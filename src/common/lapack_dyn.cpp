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

#include "clap/lapack_dyn.hpp"
#include "clap/blas/types.hpp"
#include <dlfcn.h>
#include <iostream>
#include <mutex>

namespace clap {
namespace dyn_lapack {

lapack_int (*p_LAPACKE_sgetrf)(int, lapack_int, lapack_int, float *, lapack_int,
                               lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_dgetrf)(int, lapack_int, lapack_int, double *,
                               lapack_int, lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_cgetrf)(int, lapack_int, lapack_int,
                               lapack_complex_float *, lapack_int,
                               lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_zgetrf)(int, lapack_int, lapack_int,
                               lapack_complex_double *, lapack_int,
                               lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_sgetrs)(int, char, lapack_int, lapack_int, const float *,
                               lapack_int, const lapack_int *, float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dgetrs)(int, char, lapack_int, lapack_int,
                               const double *, lapack_int, const lapack_int *,
                               double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cgetrs)(int, char, lapack_int, lapack_int,
                               const lapack_complex_float *, lapack_int,
                               const lapack_int *, lapack_complex_float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zgetrs)(int, char, lapack_int, lapack_int,
                               const lapack_complex_double *, lapack_int,
                               const lapack_int *, lapack_complex_double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_sgesv)(int, lapack_int, lapack_int, float *, lapack_int,
                              lapack_int *, float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dgesv)(int, lapack_int, lapack_int, double *, lapack_int,
                              lapack_int *, double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cgesv)(int, lapack_int, lapack_int,
                              lapack_complex_float *, lapack_int, lapack_int *,
                              lapack_complex_float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zgesv)(int, lapack_int, lapack_int,
                              lapack_complex_double *, lapack_int, lapack_int *,
                              lapack_complex_double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_sgetri)(int, lapack_int, float *, lapack_int,
                               const lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_dgetri)(int, lapack_int, double *, lapack_int,
                               const lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_cgetri)(int, lapack_int, lapack_complex_float *,
                               lapack_int, const lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_zgetri)(int, lapack_int, lapack_complex_double *,
                               lapack_int, const lapack_int *) = nullptr;
lapack_int (*p_LAPACKE_spotrf)(int, char, lapack_int, float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dpotrf)(int, char, lapack_int, double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cpotrf)(int, char, lapack_int, lapack_complex_float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zpotrf)(int, char, lapack_int, lapack_complex_double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_spotri)(int, char, lapack_int, float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dpotri)(int, char, lapack_int, double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cpotri)(int, char, lapack_int, lapack_complex_float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zpotri)(int, char, lapack_int, lapack_complex_double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_spotrs)(int, char, lapack_int, lapack_int, const float *,
                               lapack_int, float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dpotrs)(int, char, lapack_int, lapack_int,
                               const double *, lapack_int, double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cpotrs)(int, char, lapack_int, lapack_int,
                               const lapack_complex_float *, lapack_int,
                               lapack_complex_float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zpotrs)(int, char, lapack_int, lapack_int,
                               const lapack_complex_double *, lapack_int,
                               lapack_complex_double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_sposv)(int, char, lapack_int, lapack_int, float *,
                              lapack_int, float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dposv)(int, char, lapack_int, lapack_int, double *,
                              lapack_int, double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cposv)(int, char, lapack_int, lapack_int,
                              lapack_complex_float *, lapack_int,
                              lapack_complex_float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zposv)(int, char, lapack_int, lapack_int,
                              lapack_complex_double *, lapack_int,
                              lapack_complex_double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_sgeqrf)(int, lapack_int, lapack_int, float *, lapack_int,
                               float *) = nullptr;
lapack_int (*p_LAPACKE_dgeqrf)(int, lapack_int, lapack_int, double *,
                               lapack_int, double *) = nullptr;
lapack_int (*p_LAPACKE_cgeqrf)(int, lapack_int, lapack_int,
                               lapack_complex_float *, lapack_int,
                               lapack_complex_float *) = nullptr;
lapack_int (*p_LAPACKE_zgeqrf)(int, lapack_int, lapack_int,
                               lapack_complex_double *, lapack_int,
                               lapack_complex_double *) = nullptr;
lapack_int (*p_LAPACKE_sorgqr)(int, lapack_int, lapack_int, lapack_int, float *,
                               lapack_int, const float *) = nullptr;
lapack_int (*p_LAPACKE_dorgqr)(int, lapack_int, lapack_int, lapack_int,
                               double *, lapack_int, const double *) = nullptr;
lapack_int (*p_LAPACKE_cungqr)(int, lapack_int, lapack_int, lapack_int,
                               lapack_complex_float *, lapack_int,
                               const lapack_complex_float *) = nullptr;
lapack_int (*p_LAPACKE_zungqr)(int, lapack_int, lapack_int, lapack_int,
                               lapack_complex_double *, lapack_int,
                               const lapack_complex_double *) = nullptr;
lapack_int (*p_LAPACKE_sgels)(int, char, lapack_int, lapack_int, lapack_int,
                              float *, lapack_int, float *,
                              lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dgels)(int, char, lapack_int, lapack_int, lapack_int,
                              double *, lapack_int, double *,
                              lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cgels)(int, char, lapack_int, lapack_int, lapack_int,
                              lapack_complex_float *, lapack_int,
                              lapack_complex_float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zgels)(int, char, lapack_int, lapack_int, lapack_int,
                              lapack_complex_double *, lapack_int,
                              lapack_complex_double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_ssyev)(int, char, char, lapack_int, float *, lapack_int,
                              float *) = nullptr;
lapack_int (*p_LAPACKE_dsyev)(int, char, char, lapack_int, double *, lapack_int,
                              double *) = nullptr;
lapack_int (*p_LAPACKE_cheev)(int, char, char, lapack_int,
                              lapack_complex_float *, lapack_int,
                              float *) = nullptr;
lapack_int (*p_LAPACKE_zheev)(int, char, char, lapack_int,
                              lapack_complex_double *, lapack_int,
                              double *) = nullptr;
lapack_int (*p_LAPACKE_sgeev)(int, char, char, lapack_int, float *, lapack_int,
                              float *, float *, float *, lapack_int, float *,
                              lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dgeev)(int, char, char, lapack_int, double *, lapack_int,
                              double *, double *, double *, lapack_int,
                              double *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_cgeev)(int, char, char, lapack_int,
                              lapack_complex_float *, lapack_int,
                              lapack_complex_float *, lapack_complex_float *,
                              lapack_int, lapack_complex_float *,
                              lapack_int) = nullptr;
lapack_int (*p_LAPACKE_zgeev)(int, char, char, lapack_int,
                              lapack_complex_double *, lapack_int,
                              lapack_complex_double *, lapack_complex_double *,
                              lapack_int, lapack_complex_double *,
                              lapack_int) = nullptr;
lapack_int (*p_LAPACKE_sgesvd)(int, char, char, lapack_int, lapack_int, float *,
                               lapack_int, float *, float *, lapack_int,
                               float *, lapack_int, float *) = nullptr;
lapack_int (*p_LAPACKE_dgesvd)(int, char, char, lapack_int, lapack_int,
                               double *, lapack_int, double *, double *,
                               lapack_int, double *, lapack_int,
                               double *) = nullptr;
lapack_int (*p_LAPACKE_cgesvd)(int, char, char, lapack_int, lapack_int,
                               lapack_complex_float *, lapack_int, float *,
                               lapack_complex_float *, lapack_int,
                               lapack_complex_float *, lapack_int,
                               float *) = nullptr;
lapack_int (*p_LAPACKE_zgesvd)(int, char, char, lapack_int, lapack_int,
                               lapack_complex_double *, lapack_int, double *,
                               lapack_complex_double *, lapack_int,
                               lapack_complex_double *, lapack_int,
                               double *) = nullptr;
lapack_int (*p_LAPACKE_strtrs)(int, char, char, char, lapack_int, lapack_int,
                               const float *, lapack_int, float *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dtrtrs)(int, char, char, char, lapack_int, lapack_int,
                               const double *, lapack_int, double *,
                               lapack_int) = nullptr;
lapack_int (*p_LAPACKE_ctrtrs)(int, char, char, char, lapack_int, lapack_int,
                               const lapack_complex_float *, lapack_int,
                               lapack_complex_float *, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_ztrtrs)(int, char, char, char, lapack_int, lapack_int,
                               const lapack_complex_double *, lapack_int,
                               lapack_complex_double *, lapack_int) = nullptr;

cusolverStatus_t (*p_cusolverDnCreate)(cusolverDnHandle_t *) = nullptr;
cusolverStatus_t (*p_cusolverDnDestroy)(cusolverDnHandle_t) = nullptr;
cusolverStatus_t (*p_cusolverDnSgetrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  float *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgetrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  double *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgetrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgetrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuDoubleComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSgetrf)(cusolverDnHandle_t, int, int, float *,
                                       int, float *, int *, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgetrf)(cusolverDnHandle_t, int, int, double *,
                                       int, double *, int *, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgetrf)(cusolverDnHandle_t, int, int,
                                       cuComplex *, int, cuComplex *, int *,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgetrf)(cusolverDnHandle_t, int, int,
                                       cuDoubleComplex *, int,
                                       cuDoubleComplex *, int *,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSgetrs)(cusolverDnHandle_t, int, int, int,
                                       const float *, int, const int *, float *,
                                       int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgetrs)(cusolverDnHandle_t, int, int, int,
                                       const double *, int, const int *,
                                       double *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgetrs)(cusolverDnHandle_t, int, int, int,
                                       const cuComplex *, int, const int *,
                                       cuComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgetrs)(cusolverDnHandle_t, int, int, int,
                                       const cuDoubleComplex *, int,
                                       const int *, cuDoubleComplex *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSpotrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  float *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDpotrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  double *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCpotrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZpotrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuDoubleComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSpotrf)(cusolverDnHandle_t, int, int, float *,
                                       int, float *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDpotrf)(cusolverDnHandle_t, int, int, double *,
                                       int, double *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCpotrf)(cusolverDnHandle_t, int, int,
                                       cuComplex *, int, cuComplex *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZpotrf)(cusolverDnHandle_t, int, int,
                                       cuDoubleComplex *, int,
                                       cuDoubleComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSpotri_bufferSize)(cusolverDnHandle_t, int, int,
                                                  float *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDpotri_bufferSize)(cusolverDnHandle_t, int, int,
                                                  double *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCpotri_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZpotri_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuDoubleComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSpotri)(cusolverDnHandle_t, int, int, float *,
                                       int, float *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDpotri)(cusolverDnHandle_t, int, int, double *,
                                       int, double *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCpotri)(cusolverDnHandle_t, int, int,
                                       cuComplex *, int, cuComplex *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZpotri)(cusolverDnHandle_t, int, int,
                                       cuDoubleComplex *, int,
                                       cuDoubleComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSpotrs)(cusolverDnHandle_t, int, int, int,
                                       const float *, int, float *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDpotrs)(cusolverDnHandle_t, int, int, int,
                                       const double *, int, double *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCpotrs)(cusolverDnHandle_t, int, int, int,
                                       const cuComplex *, int, cuComplex *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZpotrs)(cusolverDnHandle_t, int, int, int,
                                       const cuDoubleComplex *, int,
                                       cuDoubleComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSgesvd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgesvd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgesvd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgesvd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSgesvd)(cusolverDnHandle_t, char, char, int, int,
                                       float *, int, float *, float *, int,
                                       float *, int, float *, int, float *,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgesvd)(cusolverDnHandle_t, char, char, int, int,
                                       double *, int, double *, double *, int,
                                       double *, int, double *, int, double *,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgesvd)(cusolverDnHandle_t, char, char, int, int,
                                       cuComplex *, int, float *, cuComplex *,
                                       int, cuComplex *, int, cuComplex *, int,
                                       float *, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgesvd)(cusolverDnHandle_t, char, char, int, int,
                                       cuDoubleComplex *, int, double *,
                                       cuDoubleComplex *, int,
                                       cuDoubleComplex *, int,
                                       cuDoubleComplex *, int, double *,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSsyevd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const float *, int,
                                                  const float *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDsyevd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const double *, int,
                                                  const double *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCheevd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const cuComplex *, int,
                                                  const float *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZheevd_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const cuDoubleComplex *,
                                                  int, const double *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSsyevd)(cusolverDnHandle_t, int, int, int,
                                       float *, int, float *, float *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDsyevd)(cusolverDnHandle_t, int, int, int,
                                       double *, int, double *, double *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCheevd)(cusolverDnHandle_t, int, int, int,
                                       cuComplex *, int, float *, cuComplex *,
                                       int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZheevd)(cusolverDnHandle_t, int, int, int,
                                       cuDoubleComplex *, int, double *,
                                       cuDoubleComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSgeqrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  float *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgeqrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  double *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgeqrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgeqrf_bufferSize)(cusolverDnHandle_t, int, int,
                                                  cuDoubleComplex *, int,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSgeqrf)(cusolverDnHandle_t, int, int, float *,
                                       int, float *, float *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDgeqrf)(cusolverDnHandle_t, int, int, double *,
                                       int, double *, double *, int,
                                       int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCgeqrf)(cusolverDnHandle_t, int, int,
                                       cuComplex *, int, cuComplex *,
                                       cuComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZgeqrf)(cusolverDnHandle_t, int, int,
                                       cuDoubleComplex *, int,
                                       cuDoubleComplex *, cuDoubleComplex *,
                                       int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSorgqr_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const float *, int,
                                                  const float *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDorgqr_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const double *, int,
                                                  const double *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCungqr_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const cuComplex *, int,
                                                  const cuComplex *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZungqr_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, const cuDoubleComplex *,
                                                  int, const cuDoubleComplex *,
                                                  int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSorgqr)(cusolverDnHandle_t, int, int, int,
                                       float *, int, const float *, float *,
                                       int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDorgqr)(cusolverDnHandle_t, int, int, int,
                                       double *, int, const double *, double *,
                                       int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCungqr)(cusolverDnHandle_t, int, int, int,
                                       cuComplex *, int, const cuComplex *,
                                       cuComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZungqr)(cusolverDnHandle_t, int, int, int,
                                       cuDoubleComplex *, int,
                                       const cuDoubleComplex *,
                                       cuDoubleComplex *, int, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnSSgels_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, float *, int, float *,
                                                  int, float *, int, void *,
                                                  size_t *) = nullptr;
cusolverStatus_t (*p_cusolverDnDDgels_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, double *, int, double *,
                                                  int, double *, int, void *,
                                                  size_t *) = nullptr;
cusolverStatus_t (*p_cusolverDnCCgels_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, cuComplex *, int,
                                                  cuComplex *, int, cuComplex *,
                                                  int, void *,
                                                  size_t *) = nullptr;
cusolverStatus_t (*p_cusolverDnZZgels_bufferSize)(cusolverDnHandle_t, int, int,
                                                  int, cuDoubleComplex *, int,
                                                  cuDoubleComplex *, int,
                                                  cuDoubleComplex *, int,
                                                  void *, size_t *) = nullptr;

cusolverStatus_t (*p_cusolverDnSSgels)(cusolverDnHandle_t, int, int, int, float *, int, float *, int, float *, int, void *, size_t, int *, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnDDgels)(cusolverDnHandle_t, int, int, int, double *, int, double *, int, double *, int, void *, size_t, int *, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnCCgels)(cusolverDnHandle_t, int, int, int, cuComplex *, int, cuComplex *, int, cuComplex *, int, void *, size_t, int *, int *) = nullptr;
cusolverStatus_t (*p_cusolverDnZZgels)(cusolverDnHandle_t, int, int, int, cuDoubleComplex *, int, cuDoubleComplex *, int, cuDoubleComplex *, int, void *, size_t, int *, int *) = nullptr;


int (*p_rocsolver_sgetrf)(rocsolver_handle, rocblas_int, rocblas_int, float *,
                          rocblas_int, rocblas_int *, rocblas_int *) = nullptr;
int (*p_rocsolver_dgetrf)(rocsolver_handle, rocblas_int, rocblas_int, double *,
                          rocblas_int, rocblas_int *, rocblas_int *) = nullptr;
int (*p_rocsolver_cgetrf)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_float_complex *, rocblas_int, rocblas_int *,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_zgetrf)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_double_complex *, rocblas_int, rocblas_int *,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_sgetrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const float *, rocblas_int, const rocblas_int *,
                          float *, rocblas_int) = nullptr;
int (*p_rocsolver_dgetrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const double *, rocblas_int, const rocblas_int *,
                          double *, rocblas_int) = nullptr;
int (*p_rocsolver_cgetrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const rocblas_float_complex *, rocblas_int,
                          const rocblas_int *, rocblas_float_complex *,
                          rocblas_int) = nullptr;
int (*p_rocsolver_zgetrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const rocblas_double_complex *, rocblas_int,
                          const rocblas_int *, rocblas_double_complex *,
                          rocblas_int) = nullptr;
int (*p_rocsolver_sgesv)(rocsolver_handle, rocblas_int, rocblas_int, float *,
                         rocblas_int, rocblas_int *, float *, rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_dgesv)(rocsolver_handle, rocblas_int, rocblas_int, double *,
                         rocblas_int, rocblas_int *, double *, rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_cgesv)(rocsolver_handle, rocblas_int, rocblas_int,
                         rocblas_float_complex *, rocblas_int, rocblas_int *,
                         rocblas_float_complex *, rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_zgesv)(rocsolver_handle, rocblas_int, rocblas_int,
                         rocblas_double_complex *, rocblas_int, rocblas_int *,
                         rocblas_double_complex *, rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_sgetri)(rocsolver_handle, rocblas_int, float *, rocblas_int,
                          const rocblas_int *, rocblas_int *) = nullptr;
int (*p_rocsolver_dgetri)(rocsolver_handle, rocblas_int, double *, rocblas_int,
                          const rocblas_int *, rocblas_int *) = nullptr;
int (*p_rocsolver_cgetri)(rocsolver_handle, rocblas_int,
                          rocblas_float_complex *, rocblas_int,
                          const rocblas_int *, rocblas_int *) = nullptr;
int (*p_rocsolver_zgetri)(rocsolver_handle, rocblas_int,
                          rocblas_double_complex *, rocblas_int,
                          const rocblas_int *, rocblas_int *) = nullptr;
int (*p_rocsolver_spotrf)(rocsolver_handle, int, rocblas_int, float *,
                          rocblas_int, rocblas_int *) = nullptr;
int (*p_rocsolver_dpotrf)(rocsolver_handle, int, rocblas_int, double *,
                          rocblas_int, rocblas_int *) = nullptr;
int (*p_rocsolver_cpotrf)(rocsolver_handle, int, rocblas_int,
                          rocblas_float_complex *, rocblas_int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_zpotrf)(rocsolver_handle, int, rocblas_int,
                          rocblas_double_complex *, rocblas_int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_spotri)(rocsolver_handle, int, rocblas_int, float *,
                          rocblas_int, rocblas_int *) = nullptr;
int (*p_rocsolver_dpotri)(rocsolver_handle, int, rocblas_int, double *,
                          rocblas_int, rocblas_int *) = nullptr;
int (*p_rocsolver_cpotri)(rocsolver_handle, int, rocblas_int,
                          rocblas_float_complex *, rocblas_int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_zpotri)(rocsolver_handle, int, rocblas_int,
                          rocblas_double_complex *, rocblas_int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_spotrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const float *, rocblas_int, float *,
                          rocblas_int) = nullptr;
int (*p_rocsolver_dpotrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const double *, rocblas_int, double *,
                          rocblas_int) = nullptr;
int (*p_rocsolver_cpotrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const rocblas_float_complex *, rocblas_int,
                          rocblas_float_complex *, rocblas_int) = nullptr;
int (*p_rocsolver_zpotrs)(rocsolver_handle, int, rocblas_int, rocblas_int,
                          const rocblas_double_complex *, rocblas_int,
                          rocblas_double_complex *, rocblas_int) = nullptr;
int (*p_rocsolver_sgesvd)(rocsolver_handle, int, int, rocblas_int, rocblas_int,
                          float *, rocblas_int, float *, float *, rocblas_int,
                          float *, rocblas_int, float *, int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_dgesvd)(rocsolver_handle, int, int, rocblas_int, rocblas_int,
                          double *, rocblas_int, double *, double *,
                          rocblas_int, double *, rocblas_int, double *, int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_cgesvd)(rocsolver_handle, int, int, rocblas_int, rocblas_int,
                          rocblas_float_complex *, rocblas_int, float *,
                          rocblas_float_complex *, rocblas_int,
                          rocblas_float_complex *, rocblas_int, float *, int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_zgesvd)(rocsolver_handle, int, int, rocblas_int, rocblas_int,
                          rocblas_double_complex *, rocblas_int, double *,
                          rocblas_double_complex *, rocblas_int,
                          rocblas_double_complex *, rocblas_int, double *, int,
                          rocblas_int *) = nullptr;
int (*p_rocsolver_ssyev)(rocsolver_handle, int, int, rocblas_int, float *,
                         rocblas_int, float *, float *,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_dsyev)(rocsolver_handle, int, int, rocblas_int, double *,
                         rocblas_int, double *, double *,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_cheev)(rocsolver_handle, int, int, rocblas_int,
                         rocblas_float_complex *, rocblas_int, float *, float *,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_zheev)(rocsolver_handle, int, int, rocblas_int,
                         rocblas_double_complex *, rocblas_int, double *,
                         double *, rocblas_int *) = nullptr;
int (*p_rocsolver_sgeqrf)(rocsolver_handle, rocblas_int, rocblas_int, float *,
                          rocblas_int, float *) = nullptr;
int (*p_rocsolver_dgeqrf)(rocsolver_handle, rocblas_int, rocblas_int, double *,
                          rocblas_int, double *) = nullptr;
int (*p_rocsolver_cgeqrf)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_float_complex *, rocblas_int,
                          rocblas_float_complex *) = nullptr;
int (*p_rocsolver_zgeqrf)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_double_complex *, rocblas_int,
                          rocblas_double_complex *) = nullptr;
int (*p_rocsolver_sorgqr)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_int, float *, rocblas_int, float *) = nullptr;
int (*p_rocsolver_dorgqr)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_int, double *, rocblas_int,
                          double *) = nullptr;
int (*p_rocsolver_cungqr)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_int, rocblas_float_complex *, rocblas_int,
                          rocblas_float_complex *) = nullptr;
int (*p_rocsolver_zungqr)(rocsolver_handle, rocblas_int, rocblas_int,
                          rocblas_int, rocblas_double_complex *, rocblas_int,
                          rocblas_double_complex *) = nullptr;
int (*p_rocsolver_sgels)(rocsolver_handle, rocblas_operation, const rocblas_int,
                         const rocblas_int, const rocblas_int, float *,
                         const rocblas_int, float *, const rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_dgels)(rocsolver_handle, rocblas_operation, const rocblas_int,
                         const rocblas_int, const rocblas_int, double *,
                         const rocblas_int, double *, const rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_cgels)(rocsolver_handle, rocblas_operation, const rocblas_int,
                         const rocblas_int, const rocblas_int,
                         rocblas_float_complex *, const rocblas_int,
                         rocblas_float_complex *, const rocblas_int,
                         rocblas_int *) = nullptr;
int (*p_rocsolver_zgels)(rocsolver_handle, rocblas_operation, const rocblas_int,
                         const rocblas_int, const rocblas_int,
                         rocblas_double_complex *, const rocblas_int,
                         rocblas_double_complex *, const rocblas_int,
                         rocblas_int *) = nullptr;

#define LOAD_SYM(handle, ptr, name)                                            \
  do {                                                                         \
    ptr = reinterpret_cast<decltype(ptr)>(dlsym(handle, #name));               \
    if (!ptr) {                                                                \
      std::cerr << "[CLAP LAPACK] Warning: symbol " #name << " not found\n";   \
    }                                                                          \
  } while (0)

bool loadOpenBLASLapack() {
  static bool loaded = false;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (loaded)
    return true;

  void *handle = nullptr;
  const char *libs[] = {"libopenblas.so",   "libopenblas.so.0",
                        "libopenblas64.so", "liblapacke.so",
                        "liblapacke.so.3",  nullptr};

  for (int i = 0; libs[i]; ++i) {
    handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
      std::cout << "[CLAP LAPACK] Loaded " << libs[i] << "\n";
      break;
    }
  }

  if (!handle) {
    std::cerr << "[CLAP LAPACK] Failed to load OpenBLAS LAPACK: " << dlerror()
              << "\n";
    return false;
  }

  LOAD_SYM(handle, p_LAPACKE_sgetrf, LAPACKE_sgetrf);
  LOAD_SYM(handle, p_LAPACKE_dgetrf, LAPACKE_dgetrf);
  LOAD_SYM(handle, p_LAPACKE_cgetrf, LAPACKE_cgetrf);
  LOAD_SYM(handle, p_LAPACKE_zgetrf, LAPACKE_zgetrf);
  LOAD_SYM(handle, p_LAPACKE_sgetrs, LAPACKE_sgetrs);
  LOAD_SYM(handle, p_LAPACKE_dgetrs, LAPACKE_dgetrs);
  LOAD_SYM(handle, p_LAPACKE_cgetrs, LAPACKE_cgetrs);
  LOAD_SYM(handle, p_LAPACKE_zgetrs, LAPACKE_zgetrs);
  LOAD_SYM(handle, p_LAPACKE_sgesv, LAPACKE_sgesv);
  LOAD_SYM(handle, p_LAPACKE_dgesv, LAPACKE_dgesv);
  LOAD_SYM(handle, p_LAPACKE_cgesv, LAPACKE_cgesv);
  LOAD_SYM(handle, p_LAPACKE_zgesv, LAPACKE_zgesv);
  LOAD_SYM(handle, p_LAPACKE_sgetri, LAPACKE_sgetri);
  LOAD_SYM(handle, p_LAPACKE_dgetri, LAPACKE_dgetri);
  LOAD_SYM(handle, p_LAPACKE_cgetri, LAPACKE_cgetri);
  LOAD_SYM(handle, p_LAPACKE_zgetri, LAPACKE_zgetri);
  LOAD_SYM(handle, p_LAPACKE_spotrf, LAPACKE_spotrf);
  LOAD_SYM(handle, p_LAPACKE_dpotrf, LAPACKE_dpotrf);
  LOAD_SYM(handle, p_LAPACKE_cpotrf, LAPACKE_cpotrf);
  LOAD_SYM(handle, p_LAPACKE_zpotrf, LAPACKE_zpotrf);
  LOAD_SYM(handle, p_LAPACKE_spotri, LAPACKE_spotri);
  LOAD_SYM(handle, p_LAPACKE_dpotri, LAPACKE_dpotri);
  LOAD_SYM(handle, p_LAPACKE_cpotri, LAPACKE_cpotri);
  LOAD_SYM(handle, p_LAPACKE_zpotri, LAPACKE_zpotri);
  LOAD_SYM(handle, p_LAPACKE_spotrs, LAPACKE_spotrs);
  LOAD_SYM(handle, p_LAPACKE_dpotrs, LAPACKE_dpotrs);
  LOAD_SYM(handle, p_LAPACKE_cpotrs, LAPACKE_cpotrs);
  LOAD_SYM(handle, p_LAPACKE_zpotrs, LAPACKE_zpotrs);
  LOAD_SYM(handle, p_LAPACKE_sposv, LAPACKE_sposv);
  LOAD_SYM(handle, p_LAPACKE_dposv, LAPACKE_dposv);
  LOAD_SYM(handle, p_LAPACKE_cposv, LAPACKE_cposv);
  LOAD_SYM(handle, p_LAPACKE_zposv, LAPACKE_zposv);
  LOAD_SYM(handle, p_LAPACKE_sgeqrf, LAPACKE_sgeqrf);
  LOAD_SYM(handle, p_LAPACKE_dgeqrf, LAPACKE_dgeqrf);
  LOAD_SYM(handle, p_LAPACKE_cgeqrf, LAPACKE_cgeqrf);
  LOAD_SYM(handle, p_LAPACKE_zgeqrf, LAPACKE_zgeqrf);
  LOAD_SYM(handle, p_LAPACKE_sorgqr, LAPACKE_sorgqr);
  LOAD_SYM(handle, p_LAPACKE_dorgqr, LAPACKE_dorgqr);
  LOAD_SYM(handle, p_LAPACKE_cungqr, LAPACKE_cungqr);
  LOAD_SYM(handle, p_LAPACKE_zungqr, LAPACKE_zungqr);
  LOAD_SYM(handle, p_LAPACKE_sgels, LAPACKE_sgels);
  LOAD_SYM(handle, p_LAPACKE_dgels, LAPACKE_dgels);
  LOAD_SYM(handle, p_LAPACKE_cgels, LAPACKE_cgels);
  LOAD_SYM(handle, p_LAPACKE_zgels, LAPACKE_zgels);
  LOAD_SYM(handle, p_LAPACKE_ssyev, LAPACKE_ssyev);
  LOAD_SYM(handle, p_LAPACKE_dsyev, LAPACKE_dsyev);
  LOAD_SYM(handle, p_LAPACKE_cheev, LAPACKE_cheev);
  LOAD_SYM(handle, p_LAPACKE_zheev, LAPACKE_zheev);
  LOAD_SYM(handle, p_LAPACKE_sgeev, LAPACKE_sgeev);
  LOAD_SYM(handle, p_LAPACKE_dgeev, LAPACKE_dgeev);
  LOAD_SYM(handle, p_LAPACKE_cgeev, LAPACKE_cgeev);
  LOAD_SYM(handle, p_LAPACKE_zgeev, LAPACKE_zgeev);
  LOAD_SYM(handle, p_LAPACKE_sgesvd, LAPACKE_sgesvd);
  LOAD_SYM(handle, p_LAPACKE_dgesvd, LAPACKE_dgesvd);
  LOAD_SYM(handle, p_LAPACKE_cgesvd, LAPACKE_cgesvd);
  LOAD_SYM(handle, p_LAPACKE_zgesvd, LAPACKE_zgesvd);
  LOAD_SYM(handle, p_LAPACKE_strtrs, LAPACKE_strtrs);
  LOAD_SYM(handle, p_LAPACKE_dtrtrs, LAPACKE_dtrtrs);
  LOAD_SYM(handle, p_LAPACKE_ctrtrs, LAPACKE_ctrtrs);
  LOAD_SYM(handle, p_LAPACKE_ztrtrs, LAPACKE_ztrtrs);

  loaded = true;
  return true;
}

bool loadCuSolver() {
  static bool loaded = false;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (loaded)
    return true;

  void *handle = nullptr;
  const char *libs[] = {"libcusolver.so", "libcusolver.so.11",
                        "libcusolver.so.12", nullptr};

  for (int i = 0; libs[i]; ++i) {
    handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
      std::cout << "[CLAP LAPACK] Loaded " << libs[i] << "\n";
      break;
    }
  }

  if (!handle) {
    std::cerr << "[CLAP LAPACK] Failed to load cuSOLVER: " << dlerror() << "\n";
    return false;
  }

  LOAD_SYM(handle, p_cusolverDnCreate, cusolverDnCreate);
  LOAD_SYM(handle, p_cusolverDnDestroy, cusolverDnDestroy);
  LOAD_SYM(handle, p_cusolverDnSgetrf_bufferSize, cusolverDnSgetrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDgetrf_bufferSize, cusolverDnDgetrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCgetrf_bufferSize, cusolverDnCgetrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZgetrf_bufferSize, cusolverDnZgetrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSgetrf, cusolverDnSgetrf);
  LOAD_SYM(handle, p_cusolverDnDgetrf, cusolverDnDgetrf);
  LOAD_SYM(handle, p_cusolverDnCgetrf, cusolverDnCgetrf);
  LOAD_SYM(handle, p_cusolverDnZgetrf, cusolverDnZgetrf);
  LOAD_SYM(handle, p_cusolverDnSgetrs, cusolverDnSgetrs);
  LOAD_SYM(handle, p_cusolverDnDgetrs, cusolverDnDgetrs);
  LOAD_SYM(handle, p_cusolverDnCgetrs, cusolverDnCgetrs);
  LOAD_SYM(handle, p_cusolverDnZgetrs, cusolverDnZgetrs);
  LOAD_SYM(handle, p_cusolverDnSpotrf_bufferSize, cusolverDnSpotrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDpotrf_bufferSize, cusolverDnDpotrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCpotrf_bufferSize, cusolverDnCpotrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZpotrf_bufferSize, cusolverDnZpotrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSpotrf, cusolverDnSpotrf);
  LOAD_SYM(handle, p_cusolverDnDpotrf, cusolverDnDpotrf);
  LOAD_SYM(handle, p_cusolverDnCpotrf, cusolverDnCpotrf);
  LOAD_SYM(handle, p_cusolverDnZpotrf, cusolverDnZpotrf);
  LOAD_SYM(handle, p_cusolverDnSpotri_bufferSize, cusolverDnSpotri_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDpotri_bufferSize, cusolverDnDpotri_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCpotri_bufferSize, cusolverDnCpotri_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZpotri_bufferSize, cusolverDnZpotri_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSpotri, cusolverDnSpotri);
  LOAD_SYM(handle, p_cusolverDnDpotri, cusolverDnDpotri);

  LOAD_SYM(handle, p_cusolverDnCpotri, cusolverDnCpotri);
  LOAD_SYM(handle, p_cusolverDnZpotri, cusolverDnZpotri);
  LOAD_SYM(handle, p_cusolverDnSpotrs, cusolverDnSpotrs);
  LOAD_SYM(handle, p_cusolverDnDpotrs, cusolverDnDpotrs);
  LOAD_SYM(handle, p_cusolverDnCpotrs, cusolverDnCpotrs);
  LOAD_SYM(handle, p_cusolverDnZpotrs, cusolverDnZpotrs);
  LOAD_SYM(handle, p_cusolverDnSgesvd_bufferSize, cusolverDnSgesvd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDgesvd_bufferSize, cusolverDnDgesvd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCgesvd_bufferSize, cusolverDnCgesvd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZgesvd_bufferSize, cusolverDnZgesvd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSgesvd, cusolverDnSgesvd);
  LOAD_SYM(handle, p_cusolverDnDgesvd, cusolverDnDgesvd);
  LOAD_SYM(handle, p_cusolverDnCgesvd, cusolverDnCgesvd);
  LOAD_SYM(handle, p_cusolverDnZgesvd, cusolverDnZgesvd);
  LOAD_SYM(handle, p_cusolverDnSsyevd_bufferSize, cusolverDnSsyevd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDsyevd_bufferSize, cusolverDnDsyevd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCheevd_bufferSize, cusolverDnCheevd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZheevd_bufferSize, cusolverDnZheevd_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSsyevd, cusolverDnSsyevd);
  LOAD_SYM(handle, p_cusolverDnDsyevd, cusolverDnDsyevd);
  LOAD_SYM(handle, p_cusolverDnCheevd, cusolverDnCheevd);
  LOAD_SYM(handle, p_cusolverDnZheevd, cusolverDnZheevd);
  LOAD_SYM(handle, p_cusolverDnSgeqrf_bufferSize, cusolverDnSgeqrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCgeqrf_bufferSize, cusolverDnCgeqrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZgeqrf_bufferSize, cusolverDnZgeqrf_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSgeqrf, cusolverDnSgeqrf);
  LOAD_SYM(handle, p_cusolverDnDgeqrf, cusolverDnDgeqrf);
  LOAD_SYM(handle, p_cusolverDnCgeqrf, cusolverDnCgeqrf);
  LOAD_SYM(handle, p_cusolverDnZgeqrf, cusolverDnZgeqrf);
  LOAD_SYM(handle, p_cusolverDnSorgqr_bufferSize, cusolverDnSorgqr_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDorgqr_bufferSize, cusolverDnDorgqr_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCungqr_bufferSize, cusolverDnCungqr_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZungqr_bufferSize, cusolverDnZungqr_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSorgqr, cusolverDnSorgqr);
  LOAD_SYM(handle, p_cusolverDnDorgqr, cusolverDnDorgqr);
  LOAD_SYM(handle, p_cusolverDnCungqr, cusolverDnCungqr);
  LOAD_SYM(handle, p_cusolverDnZungqr, cusolverDnZungqr);
  LOAD_SYM(handle, p_cusolverDnSSgels_bufferSize, cusolverDnSSgels_bufferSize);
  LOAD_SYM(handle, p_cusolverDnDDgels_bufferSize, cusolverDnDDgels_bufferSize);
  LOAD_SYM(handle, p_cusolverDnCCgels_bufferSize, cusolverDnCCgels_bufferSize);
  LOAD_SYM(handle, p_cusolverDnZZgels_bufferSize, cusolverDnZZgels_bufferSize);
  LOAD_SYM(handle, p_cusolverDnSSgels, cusolverDnSSgels);
  LOAD_SYM(handle, p_cusolverDnDDgels, cusolverDnDDgels);
  LOAD_SYM(handle, p_cusolverDnCCgels, cusolverDnCCgels);
  LOAD_SYM(handle, p_cusolverDnZZgels, cusolverDnZZgels);

  loaded = true;
  return true;
}

bool loadRocSolver() {
  static bool loaded = false;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (loaded)
    return true;

  void *handle = nullptr;
  const char *libs[] = {"librocsolver.so", "librocsolver.so.0", nullptr};

  for (int i = 0; libs[i]; ++i) {
    handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
      std::cout << "[CLAP LAPACK] Loaded " << libs[i] << "\n";
      break;
    }
  }

  if (!handle) {
    std::cerr << "[CLAP LAPACK] Failed to load rocSOLVER: " << dlerror()
              << "\n";
    return false;
  }

  LOAD_SYM(handle, p_rocsolver_sgetrf, rocsolver_sgetrf);
  LOAD_SYM(handle, p_rocsolver_dgetrf, rocsolver_dgetrf);
  LOAD_SYM(handle, p_rocsolver_cgetrf, rocsolver_cgetrf);
  LOAD_SYM(handle, p_rocsolver_zgetrf, rocsolver_zgetrf);
  LOAD_SYM(handle, p_rocsolver_sgetrs, rocsolver_sgetrs);
  LOAD_SYM(handle, p_rocsolver_dgetrs, rocsolver_dgetrs);
  LOAD_SYM(handle, p_rocsolver_cgetrs, rocsolver_cgetrs);
  LOAD_SYM(handle, p_rocsolver_zgetrs, rocsolver_zgetrs);
  LOAD_SYM(handle, p_rocsolver_sgesv, rocsolver_sgesv);
  LOAD_SYM(handle, p_rocsolver_dgesv, rocsolver_dgesv);
  LOAD_SYM(handle, p_rocsolver_cgesv, rocsolver_cgesv);
  LOAD_SYM(handle, p_rocsolver_zgesv, rocsolver_zgesv);
  LOAD_SYM(handle, p_rocsolver_sgetri, rocsolver_sgetri);
  LOAD_SYM(handle, p_rocsolver_dgetri, rocsolver_dgetri);
  LOAD_SYM(handle, p_rocsolver_cgetri, rocsolver_cgetri);
  LOAD_SYM(handle, p_rocsolver_zgetri, rocsolver_zgetri);
  LOAD_SYM(handle, p_rocsolver_spotrf, rocsolver_spotrf);
  LOAD_SYM(handle, p_rocsolver_dpotrf, rocsolver_dpotrf);
  LOAD_SYM(handle, p_rocsolver_cpotrf, rocsolver_cpotrf);
  LOAD_SYM(handle, p_rocsolver_zpotrf, rocsolver_zpotrf);
  LOAD_SYM(handle, p_rocsolver_spotri, rocsolver_spotri);
  LOAD_SYM(handle, p_rocsolver_dpotri, rocsolver_dpotri);

  LOAD_SYM(handle, p_rocsolver_cpotri, rocsolver_cpotri);
  LOAD_SYM(handle, p_rocsolver_zpotri, rocsolver_zpotri);
  LOAD_SYM(handle, p_rocsolver_spotrs, rocsolver_spotrs);
  LOAD_SYM(handle, p_rocsolver_dpotrs, rocsolver_dpotrs);
  LOAD_SYM(handle, p_rocsolver_cpotrs, rocsolver_cpotrs);
  LOAD_SYM(handle, p_rocsolver_zpotrs, rocsolver_zpotrs);
  LOAD_SYM(handle, p_rocsolver_sgesvd, rocsolver_sgesvd);
  LOAD_SYM(handle, p_rocsolver_dgesvd, rocsolver_dgesvd);
  LOAD_SYM(handle, p_rocsolver_cgesvd, rocsolver_cgesvd);
  LOAD_SYM(handle, p_rocsolver_zgesvd, rocsolver_zgesvd);
  LOAD_SYM(handle, p_rocsolver_ssyev, rocsolver_ssyev);
  LOAD_SYM(handle, p_rocsolver_dsyev, rocsolver_dsyev);
  LOAD_SYM(handle, p_rocsolver_cheev, rocsolver_cheev);
  LOAD_SYM(handle, p_rocsolver_zheev, rocsolver_zheev);
  LOAD_SYM(handle, p_rocsolver_sgeqrf, rocsolver_sgeqrf);
  LOAD_SYM(handle, p_rocsolver_dgeqrf, rocsolver_dgeqrf);
  LOAD_SYM(handle, p_rocsolver_cgeqrf, rocsolver_cgeqrf);
  LOAD_SYM(handle, p_rocsolver_zgeqrf, rocsolver_zgeqrf);
  LOAD_SYM(handle, p_rocsolver_sorgqr, rocsolver_sorgqr);
  LOAD_SYM(handle, p_rocsolver_dorgqr, rocsolver_dorgqr);
  LOAD_SYM(handle, p_rocsolver_cungqr, rocsolver_cungqr);
  LOAD_SYM(handle, p_rocsolver_zungqr, rocsolver_zungqr);
  LOAD_SYM(handle, p_rocsolver_sgels, rocsolver_sgels);
  LOAD_SYM(handle, p_rocsolver_dgels, rocsolver_dgels);
  LOAD_SYM(handle, p_rocsolver_cgels, rocsolver_cgels);
  LOAD_SYM(handle, p_rocsolver_zgels, rocsolver_zgels);

  loaded = true;
  return true;
}

} // namespace dyn_lapack
} // namespace clap
