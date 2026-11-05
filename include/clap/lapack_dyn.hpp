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

#pragma once
#include "clap/blas/types.hpp"
#include "clap/lapack/lapack_types.hpp"
#include <cstdint>
#include <dlfcn.h>
#include <iostream>

namespace clap {
namespace dyn_lapack {

extern lapack_int (*p_LAPACKE_sgetrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, float *a, lapack_int lda,
                                      lapack_int *ipiv);
extern lapack_int (*p_LAPACKE_dgetrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, double *a, lapack_int lda,
                                      lapack_int *ipiv);
extern lapack_int (*p_LAPACKE_cgetrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float *a,
                                      lapack_int lda, lapack_int *ipiv);
extern lapack_int (*p_LAPACKE_zgetrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double *a,
                                      lapack_int lda, lapack_int *ipiv);

extern lapack_int (*p_LAPACKE_sgetrs)(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const float *a, lapack_int lda,
                                      const lapack_int *ipiv, float *b,
                                      lapack_int ldb);
extern lapack_int (*p_LAPACKE_dgetrs)(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const double *a, lapack_int lda,
                                      const lapack_int *ipiv, double *b,
                                      lapack_int ldb);
extern lapack_int (*p_LAPACKE_cgetrs)(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float *a,
                                      lapack_int lda, const lapack_int *ipiv,
                                      lapack_complex_float *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_zgetrs)(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double *a,
                                      lapack_int lda, const lapack_int *ipiv,
                                      lapack_complex_double *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_sgesv)(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, float *a, lapack_int lda,
                                     lapack_int *ipiv, float *b,
                                     lapack_int ldb);
extern lapack_int (*p_LAPACKE_dgesv)(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, double *a, lapack_int lda,
                                     lapack_int *ipiv, double *b,
                                     lapack_int ldb);

extern lapack_int (*p_LAPACKE_cgesv)(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float *a,
                                     lapack_int lda, lapack_int *ipiv,
                                     lapack_complex_float *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_zgesv)(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double *a,
                                     lapack_int lda, lapack_int *ipiv,
                                     lapack_complex_double *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_sgetri)(int matrix_layout, lapack_int n, float *a,
                                      lapack_int lda, const lapack_int *ipiv);
extern lapack_int (*p_LAPACKE_dgetri)(int matrix_layout, lapack_int n,
                                      double *a, lapack_int lda,
                                      const lapack_int *ipiv);

extern lapack_int (*p_LAPACKE_cgetri)(int matrix_layout, lapack_int n,
                                      lapack_complex_float *a, lapack_int lda,
                                      const lapack_int *ipiv);

extern lapack_int (*p_LAPACKE_zgetri)(int matrix_layout, lapack_int n,
                                      lapack_complex_double *a, lapack_int lda,
                                      const lapack_int *ipiv);

extern lapack_int (*p_LAPACKE_spotrf)(int matrix_layout, char uplo,
                                      lapack_int n, float *a, lapack_int lda);
extern lapack_int (*p_LAPACKE_dpotrf)(int matrix_layout, char uplo,
                                      lapack_int n, double *a, lapack_int lda);
extern lapack_int (*p_LAPACKE_cpotrf)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float *a,
                                      lapack_int lda);
extern lapack_int (*p_LAPACKE_zpotrf)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double *a,
                                      lapack_int lda);

extern lapack_int (*p_LAPACKE_spotri)(int matrix_layout, char uplo,
                                      lapack_int n, float *A, lapack_int lda);

extern lapack_int (*p_LAPACKE_dpotri)(int matrix_layout, char uplo,
                                      lapack_int n, double *A, lapack_int lda);

extern lapack_int (*p_LAPACKE_cpotri)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float *A,
                                      lapack_int lda);

extern lapack_int (*p_LAPACKE_zpotri)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double *A,
                                      lapack_int lda);

extern lapack_int (*p_LAPACKE_spotrs)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float *a, lapack_int lda, float *b,
                                      lapack_int ldb);
extern lapack_int (*p_LAPACKE_dpotrs)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double *a, lapack_int lda,
                                      double *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_cpotrs)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float *a,
                                      lapack_int lda, lapack_complex_float *b,
                                      lapack_int ldb);

extern lapack_int (*p_LAPACKE_zpotrs)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double *a,
                                      lapack_int lda, lapack_complex_double *b,
                                      lapack_int ldb);

extern lapack_int (*p_LAPACKE_sposv)(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, float *a, lapack_int lda,
                                     float *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dposv)(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, double *a, lapack_int lda,
                                     double *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_cposv)(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float *a,
                                     lapack_int lda, lapack_complex_float *b,
                                     lapack_int ldb);
extern lapack_int (*p_LAPACKE_zposv)(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double *a,
                                     lapack_int lda, lapack_complex_double *b,
                                     lapack_int ldb);

extern lapack_int (*p_LAPACKE_sgeqrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, float *a, lapack_int lda,
                                      float *tau);
extern lapack_int (*p_LAPACKE_dgeqrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, double *a, lapack_int lda,
                                      double *tau);
extern lapack_int (*p_LAPACKE_cgeqrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float *a,
                                      lapack_int lda,
                                      lapack_complex_float *tau);
extern lapack_int (*p_LAPACKE_zgeqrf)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double *a,
                                      lapack_int lda,
                                      lapack_complex_double *tau);

extern lapack_int (*p_LAPACKE_sorgqr)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, float *a,
                                      lapack_int lda, const float *tau);
extern lapack_int (*p_LAPACKE_dorgqr)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, double *a,
                                      lapack_int lda, const double *tau);
extern lapack_int (*p_LAPACKE_cungqr)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_float *a, lapack_int lda,
                                      const lapack_complex_float *tau);
extern lapack_int (*p_LAPACKE_zungqr)(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_double *a, lapack_int lda,
                                      const lapack_complex_double *tau);

extern lapack_int (*p_LAPACKE_sgels)(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, float *a, lapack_int lda,
                                     float *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dgels)(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, double *a, lapack_int lda,
                                     double *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_cgels)(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float *a,
                                     lapack_int lda, lapack_complex_float *b,
                                     lapack_int ldb);
extern lapack_int (*p_LAPACKE_zgels)(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double *a,
                                     lapack_int lda, lapack_complex_double *b,
                                     lapack_int ldb);

extern lapack_int (*p_LAPACKE_ssyev)(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, float *a, lapack_int lda,
                                     float *w);
extern lapack_int (*p_LAPACKE_dsyev)(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, double *a, lapack_int lda,
                                     double *w);
extern lapack_int (*p_LAPACKE_cheev)(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, lapack_complex_float *a,
                                     lapack_int lda, float *w);
extern lapack_int (*p_LAPACKE_zheev)(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, lapack_complex_double *a,
                                     lapack_int lda, double *w);

extern lapack_int (*p_LAPACKE_sgeev)(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, float *a, lapack_int lda,
                                     float *wr, float *wi, float *vl,
                                     lapack_int ldvl, float *vr,
                                     lapack_int ldvr);
extern lapack_int (*p_LAPACKE_dgeev)(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, double *a, lapack_int lda,
                                     double *wr, double *wi, double *vl,
                                     lapack_int ldvl, double *vr,
                                     lapack_int ldvr);
extern lapack_int (*p_LAPACKE_cgeev)(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, lapack_complex_float *a,
                                     lapack_int lda, lapack_complex_float *w,
                                     lapack_complex_float *vl, lapack_int ldvl,
                                     lapack_complex_float *vr, lapack_int ldvr);
extern lapack_int (*p_LAPACKE_zgeev)(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, lapack_complex_double *a,
                                     lapack_int lda, lapack_complex_double *w,
                                     lapack_complex_double *vl, lapack_int ldvl,
                                     lapack_complex_double *vr,
                                     lapack_int ldvr);

extern lapack_int (*p_LAPACKE_sgesvd)(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n, float *a,
                                      lapack_int lda, float *s, float *u,
                                      lapack_int ldu, float *vt,
                                      lapack_int ldvt, float *superb);
extern lapack_int (*p_LAPACKE_dgesvd)(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n, double *a,
                                      lapack_int lda, double *s, double *u,
                                      lapack_int ldu, double *vt,
                                      lapack_int ldvt, double *superb);
extern lapack_int (*p_LAPACKE_cgesvd)(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_float *a, lapack_int lda,
                                      float *s, lapack_complex_float *u,
                                      lapack_int ldu, lapack_complex_float *vt,
                                      lapack_int ldvt, float *rwork);
extern lapack_int (*p_LAPACKE_zgesvd)(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_double *a, lapack_int lda,
                                      double *s, lapack_complex_double *u,
                                      lapack_int ldu, lapack_complex_double *vt,
                                      lapack_int ldvt, double *rwork);

extern lapack_int (*p_LAPACKE_strtrs)(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const float *a, lapack_int lda, float *b,
                                      lapack_int ldb);
extern lapack_int (*p_LAPACKE_dtrtrs)(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const double *a, lapack_int lda,
                                      double *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_ctrtrs)(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float *a,
                                      lapack_int lda, lapack_complex_float *b,
                                      lapack_int ldb);
extern lapack_int (*p_LAPACKE_ztrtrs)(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double *a,
                                      lapack_int lda, lapack_complex_double *b,
                                      lapack_int ldb);

extern cusolverStatus_t (*p_cusolverDnCreate)(cusolverDnHandle_t *handle);
extern cusolverStatus_t (*p_cusolverDnDestroy)(cusolverDnHandle_t handle);

extern cusolverStatus_t (*p_cusolverDnSgetrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDgetrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnCgetrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnZgetrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda,
    int *Lwork);

extern cusolverStatus_t (*p_cusolverDnSgetrf)(cusolverDnHandle_t handle, int m,
                                              int n, float *A, int lda,
                                              float *Workspace, int *devIpiv,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgetrf)(cusolverDnHandle_t handle, int m,
                                              int n, double *A, int lda,
                                              double *Workspace, int *devIpiv,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCgetrf)(cusolverDnHandle_t handle, int m,
                                              int n, cuComplex *A, int lda,
                                              cuComplex *Workspace,
                                              int *devIpiv, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZgetrf)(cusolverDnHandle_t handle, int m,
                                              int n, cuDoubleComplex *A,
                                              int lda,
                                              cuDoubleComplex *Workspace,
                                              int *devIpiv, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSgetrs)(cusolverDnHandle_t handle,
                                              int trans, int n, int nrhs,
                                              const float *A, int lda,
                                              const int *devIpiv, float *B,
                                              int ldb, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgetrs)(cusolverDnHandle_t handle,
                                              int trans, int n, int nrhs,
                                              const double *A, int lda,
                                              const int *devIpiv, double *B,
                                              int ldb, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCgetrs)(cusolverDnHandle_t handle,
                                              int trans, int n, int nrhs,
                                              const cuComplex *A, int lda,
                                              const int *devIpiv, cuComplex *B,
                                              int ldb, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZgetrs)(cusolverDnHandle_t handle,
                                              int trans, int n, int nrhs,
                                              const cuDoubleComplex *A, int lda,
                                              const int *devIpiv,
                                              cuDoubleComplex *B, int ldb,
                                              int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSpotrf_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, float *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDpotrf_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, double *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnCpotrf_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, cuComplex *A, int lda,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnZpotrf_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, cuDoubleComplex *A, int lda,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnSpotrf)(cusolverDnHandle_t handle,
                                              int uplo, int n, float *A,
                                              int lda, float *Workspace,
                                              int Lwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDpotrf)(cusolverDnHandle_t handle,
                                              int uplo, int n, double *A,
                                              int lda, double *Workspace,
                                              int Lwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCpotrf)(cusolverDnHandle_t handle,
                                              int uplo, int n, cuComplex *A,
                                              int lda, cuComplex *Workspace,
                                              int Lwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZpotrf)(cusolverDnHandle_t handle,
                                              int uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSpotri_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, float *A, int lda, int *Lwork);

extern cusolverStatus_t (*p_cusolverDnDpotri_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, double *A, int lda, int *Lwork);

extern cusolverStatus_t (*p_cusolverDnCpotri_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, cuComplex *A, int lda,
    int *Lwork);

extern cusolverStatus_t (*p_cusolverDnZpotri_bufferSize)(
    cusolverDnHandle_t handle, int uplo, int n, cuDoubleComplex *A, int lda,
    int *Lwork);

extern cusolverStatus_t (*p_cusolverDnSpotri)(cusolverDnHandle_t handle,
                                              int uplo, int n, float *A,
                                              int lda, float *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnDpotri)(cusolverDnHandle_t handle,
                                              int uplo, int n, double *A,
                                              int lda, double *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnCpotri)(cusolverDnHandle_t handle,
                                              int uplo, int n, cuComplex *A,
                                              int lda, cuComplex *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnZpotri)(cusolverDnHandle_t handle,
                                              int uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSpotrs)(cusolverDnHandle_t handle,
                                              int uplo, int n, int nrhs,
                                              const float *A, int lda, float *B,
                                              int ldb, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnDpotrs)(cusolverDnHandle_t handle,
                                              int uplo, int n, int nrhs,
                                              const double *A, int lda,
                                              double *B, int ldb, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnCpotrs)(cusolverDnHandle_t handle,
                                              int uplo, int n, int nrhs,
                                              const cuComplex *A, int lda,
                                              cuComplex *B, int ldb,
                                              int *devInfo);

extern cusolverStatus_t (*p_cusolverDnZpotrs)(cusolverDnHandle_t handle,
                                              int uplo, int n, int nrhs,
                                              const cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *B, int ldb,
                                              int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSgesvd_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDgesvd_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnCgesvd_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnZgesvd_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnSgesvd)(
    cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n, float *A,
    int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work,
    int Lwork, float *rwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgesvd)(
    cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n, double *A,
    int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work,
    int Lwork, double *rwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCgesvd)(
    cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n,
    cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *VT,
    int ldvt, cuComplex *Work, int Lwork, float *rwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZgesvd)(
    cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n,
    cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu,
    cuDoubleComplex *VT, int ldvt, cuDoubleComplex *Work, int Lwork,
    double *rwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSsyevd_bufferSize)(
    cusolverDnHandle_t handle, int jobz, int uplo, int n, const float *A,
    int lda, const float *W, int *lwork);
extern cusolverStatus_t (*p_cusolverDnDsyevd_bufferSize)(
    cusolverDnHandle_t handle, int jobz, int uplo, int n, const double *A,
    int lda, const double *W, int *lwork);
extern cusolverStatus_t (*p_cusolverDnCheevd_bufferSize)(
    cusolverDnHandle_t handle, int jobz, int uplo, int n, const cuComplex *A,
    int lda, const float *W, int *lwork);
extern cusolverStatus_t (*p_cusolverDnZheevd_bufferSize)(
    cusolverDnHandle_t handle, int jobz, int uplo, int n,
    const cuDoubleComplex *A, int lda, const double *W, int *lwork);
extern cusolverStatus_t (*p_cusolverDnSsyevd)(cusolverDnHandle_t handle,
                                              int jobz, int uplo, int n,
                                              float *A, int lda, float *W,
                                              float *work, int lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDsyevd)(cusolverDnHandle_t handle,
                                              int jobz, int uplo, int n,
                                              double *A, int lda, double *W,
                                              double *work, int lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCheevd)(cusolverDnHandle_t handle,
                                              int jobz, int uplo, int n,
                                              cuComplex *A, int lda, float *W,
                                              cuComplex *work, int lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZheevd)(cusolverDnHandle_t handle,
                                              int jobz, int uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              double *W, cuDoubleComplex *work,
                                              int lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSgeqrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDgeqrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnCgeqrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnZgeqrf_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnSgeqrf)(cusolverDnHandle_t handle, int m,
                                              int n, float *A, int lda,
                                              float *tau, float *Workspace,
                                              int Lwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgeqrf)(cusolverDnHandle_t handle, int m,
                                              int n, double *A, int lda,
                                              double *tau, double *Workspace,
                                              int Lwork, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCgeqrf)(cusolverDnHandle_t handle, int m,
                                              int n, cuComplex *A, int lda,
                                              cuComplex *tau,
                                              cuComplex *Workspace, int Lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZgeqrf)(cusolverDnHandle_t handle, int m,
                                              int n, cuDoubleComplex *A,
                                              int lda, cuDoubleComplex *tau,
                                              cuDoubleComplex *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSorgqr_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int k, const float *A, int lda,
    const float *tau, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDorgqr_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int k, const double *A, int lda,
    const double *tau, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnCungqr_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int k, const cuComplex *A, int lda,
    const cuComplex *tau, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnZungqr_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *tau, int *Lwork);
extern cusolverStatus_t (*p_cusolverDnSorgqr)(cusolverDnHandle_t handle, int m,
                                              int n, int k, float *A, int lda,
                                              const float *tau,
                                              float *Workspace, int Lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDorgqr)(cusolverDnHandle_t handle, int m,
                                              int n, int k, double *A, int lda,
                                              const double *tau,
                                              double *Workspace, int Lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCungqr)(cusolverDnHandle_t handle, int m,
                                              int n, int k, cuComplex *A,
                                              int lda, const cuComplex *tau,
                                              cuComplex *Workspace, int Lwork,
                                              int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZungqr)(cusolverDnHandle_t handle, int m,
                                              int n, int k, cuDoubleComplex *A,
                                              int lda,
                                              const cuDoubleComplex *tau,
                                              cuDoubleComplex *Workspace,
                                              int Lwork, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnSSgels_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int nrhs, float *A, int lda,
    float *B, int ldb, float *X, int ldx, void *dwork, size_t *Lwork);
extern cusolverStatus_t (*p_cusolverDnDDgels_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int nrhs, double *A, int lda,
    double *B, int ldb, double *X, int ldx, void *dwork, size_t *Lwork);
extern cusolverStatus_t (*p_cusolverDnCCgels_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *A, int lda,
    cuComplex *B, int ldb, cuComplex *X, int ldx, void *dwork, size_t *Lwork);
extern cusolverStatus_t (*p_cusolverDnZZgels_bufferSize)(
    cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *A,
    int lda, cuDoubleComplex *B, int ldb, cuDoubleComplex *X, int ldx,
    void *dwork, size_t *Lwork);

extern cusolverStatus_t (*p_cusolverDnSSgels)(cusolverDnHandle_t handle, int m, int n, int nrhs, float *A, int lda, float *B, int ldb, float *X, int ldx, void *dwork, size_t Lwork, int *niter, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDDgels)(cusolverDnHandle_t handle, int m, int n, int nrhs, double *A, int lda, double *B, int ldb, double *X, int ldx, void *dwork, size_t Lwork, int *niter, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnCCgels)(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *A, int lda, cuComplex *B, int ldb, cuComplex *X, int ldx, void *dwork, size_t Lwork, int *niter, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnZZgels)(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, cuDoubleComplex *X, int ldx, void *dwork, size_t Lwork, int *niter, int *devInfo);


extern int (*p_rocsolver_sgetrf)(rocsolver_handle handle, rocblas_int m,
                                 rocblas_int n, float *A, rocblas_int lda,
                                 rocblas_int *ipiv, rocblas_int *info);
extern int (*p_rocsolver_dgetrf)(rocsolver_handle handle, rocblas_int m,
                                 rocblas_int n, double *A, rocblas_int lda,
                                 rocblas_int *ipiv, rocblas_int *info);
extern int (*p_rocsolver_cgetrf)(rocsolver_handle handle, rocblas_int m,
                                 rocblas_int n, rocblas_float_complex *A,
                                 rocblas_int lda, rocblas_int *ipiv,
                                 rocblas_int *info);
extern int (*p_rocsolver_zgetrf)(rocsolver_handle handle, rocblas_int m,
                                 rocblas_int n, rocblas_double_complex *A,
                                 rocblas_int lda, rocblas_int *ipiv,
                                 rocblas_int *info);

extern int (*p_rocsolver_sgetrs)(rocsolver_handle handle, int trans,
                                 rocblas_int n, rocblas_int nrhs,
                                 const float *A, rocblas_int lda,
                                 const rocblas_int *ipiv, float *B,
                                 rocblas_int ldb);
extern int (*p_rocsolver_dgetrs)(rocsolver_handle handle, int trans,
                                 rocblas_int n, rocblas_int nrhs,
                                 const double *A, rocblas_int lda,
                                 const rocblas_int *ipiv, double *B,
                                 rocblas_int ldb);
extern int (*p_rocsolver_cgetrs)(rocsolver_handle handle, int trans,
                                 rocblas_int n, rocblas_int nrhs,
                                 const rocblas_float_complex *A,
                                 rocblas_int lda, const rocblas_int *ipiv,
                                 rocblas_float_complex *B, rocblas_int ldb);
extern int (*p_rocsolver_zgetrs)(rocsolver_handle handle, int trans,
                                 rocblas_int n, rocblas_int nrhs,
                                 const rocblas_double_complex *A,
                                 rocblas_int lda, const rocblas_int *ipiv,
                                 rocblas_double_complex *B, rocblas_int ldb);

extern int (*p_rocsolver_sgesv)(rocsolver_handle handle, rocblas_int n,
                                rocblas_int nrhs, float *A, rocblas_int lda,
                                rocblas_int *ipiv, float *B, rocblas_int ldb,
                                rocblas_int *info);
extern int (*p_rocsolver_dgesv)(rocsolver_handle handle, rocblas_int n,
                                rocblas_int nrhs, double *A, rocblas_int lda,
                                rocblas_int *ipiv, double *B, rocblas_int ldb,
                                rocblas_int *info);
extern int (*p_rocsolver_cgesv)(rocsolver_handle handle, rocblas_int n,
                                rocblas_int nrhs, rocblas_float_complex *A,
                                rocblas_int lda, rocblas_int *ipiv,
                                rocblas_float_complex *B, rocblas_int ldb,
                                rocblas_int *info);
extern int (*p_rocsolver_zgesv)(rocsolver_handle handle, rocblas_int n,
                                rocblas_int nrhs, rocblas_double_complex *A,
                                rocblas_int lda, rocblas_int *ipiv,
                                rocblas_double_complex *B, rocblas_int ldb,
                                rocblas_int *info);

extern int (*p_rocsolver_sgetri)(rocsolver_handle handle, rocblas_int n,
                                 float *A, rocblas_int lda,
                                 const rocblas_int *ipiv, rocblas_int *info);
extern int (*p_rocsolver_dgetri)(rocsolver_handle handle, rocblas_int n,
                                 double *A, rocblas_int lda,
                                 const rocblas_int *ipiv, rocblas_int *info);
extern int (*p_rocsolver_cgetri)(rocsolver_handle handle, rocblas_int n,
                                 rocblas_float_complex *A, rocblas_int lda,
                                 const rocblas_int *ipiv, rocblas_int *info);
extern int (*p_rocsolver_zgetri)(rocsolver_handle handle, rocblas_int n,
                                 rocblas_double_complex *A, rocblas_int lda,
                                 const rocblas_int *ipiv, rocblas_int *info);

extern int (*p_rocsolver_spotrf)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, float *A, rocblas_int lda,
                                 rocblas_int *info);
extern int (*p_rocsolver_dpotrf)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, double *A, rocblas_int lda,
                                 rocblas_int *info);
extern int (*p_rocsolver_cpotrf)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_float_complex *A,
                                 rocblas_int lda, rocblas_int *info);
extern int (*p_rocsolver_zpotrf)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_double_complex *A,
                                 rocblas_int lda, rocblas_int *info);

extern int (*p_rocsolver_spotri)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, float *A, rocblas_int lda,
                                 rocblas_int *info);

extern int (*p_rocsolver_dpotri)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, double *A, rocblas_int lda,
                                 rocblas_int *info);
extern int (*p_rocsolver_cpotri)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_float_complex *A,
                                 rocblas_int lda, rocblas_int *info);
extern int (*p_rocsolver_zpotri)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_double_complex *A,
                                 rocblas_int lda, rocblas_int *info);

extern int (*p_rocsolver_spotrs)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_int nrhs,
                                 const float *A, rocblas_int lda, float *B,
                                 rocblas_int ldb);

extern int (*p_rocsolver_dpotrs)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_int nrhs,
                                 const double *A, rocblas_int lda, double *B,
                                 rocblas_int ldb);
extern int (*p_rocsolver_cpotrs)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_int nrhs,
                                 const rocblas_float_complex *A,
                                 rocblas_int lda, rocblas_float_complex *B,
                                 rocblas_int ldb);
extern int (*p_rocsolver_zpotrs)(rocsolver_handle handle, int uplo,
                                 rocblas_int n, rocblas_int nrhs,
                                 const rocblas_double_complex *A,
                                 rocblas_int lda, rocblas_double_complex *B,
                                 rocblas_int ldb);

extern int (*p_rocsolver_sgesvd)(rocsolver_handle handle, int jobu, int jobv,
                                 rocblas_int m, rocblas_int n, float *A,
                                 rocblas_int lda, float *S, float *U,
                                 rocblas_int ldu, float *V, rocblas_int ldv,
                                 float *E, int fast_alg, rocblas_int *info);
extern int (*p_rocsolver_dgesvd)(rocsolver_handle handle, int jobu, int jobv,
                                 rocblas_int m, rocblas_int n, double *A,
                                 rocblas_int lda, double *S, double *U,
                                 rocblas_int ldu, double *V, rocblas_int ldv,
                                 double *E, int fast_alg, rocblas_int *info);
extern int (*p_rocsolver_cgesvd)(rocsolver_handle handle, int jobu, int jobv,
                                 rocblas_int m, rocblas_int n,
                                 rocblas_float_complex *A, rocblas_int lda,
                                 float *S, rocblas_float_complex *U,
                                 rocblas_int ldu, rocblas_float_complex *V,
                                 rocblas_int ldv, float *E, int fast_alg,
                                 rocblas_int *info);
extern int (*p_rocsolver_zgesvd)(rocsolver_handle handle, int jobu, int jobv,
                                 rocblas_int m, rocblas_int n,
                                 rocblas_double_complex *A, rocblas_int lda,
                                 double *S, rocblas_double_complex *U,
                                 rocblas_int ldu, rocblas_double_complex *V,
                                 rocblas_int ldv, double *E, int fast_alg,
                                 rocblas_int *info);

extern int (*p_rocsolver_ssyev)(rocsolver_handle handle, int evect, int uplo,
                                rocblas_int n, float *A, rocblas_int lda,
                                float *D, float *E, rocblas_int *info);
extern int (*p_rocsolver_dsyev)(rocsolver_handle handle, int evect, int uplo,
                                rocblas_int n, double *A, rocblas_int lda,
                                double *D, double *E, rocblas_int *info);
extern int (*p_rocsolver_cheev)(rocsolver_handle handle, int evect, int uplo,
                                rocblas_int n, rocblas_float_complex *A,
                                rocblas_int lda, float *D, float *E,
                                rocblas_int *info);
extern int (*p_rocsolver_zheev)(rocsolver_handle handle, int evect, int uplo,
                                rocblas_int n, rocblas_double_complex *A,
                                rocblas_int lda, double *D, double *E,
                                rocblas_int *info);
extern int (*p_rocsolver_sgeqrf)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, float *A,
                                 const rocblas_int lda, float *ipiv);
extern int (*p_rocsolver_dgeqrf)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, double *A,
                                 const rocblas_int lda, double *ipiv);
extern int (*p_rocsolver_cgeqrf)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, rocblas_float_complex *A,
                                 const rocblas_int lda,
                                 rocblas_float_complex *ipiv);
extern int (*p_rocsolver_zgeqrf)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, rocblas_double_complex *A,
                                 const rocblas_int lda,
                                 rocblas_double_complex *ipiv);

extern int (*p_rocsolver_sorgqr)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, const rocblas_int k,
                                 float *A, const rocblas_int lda, float *ipiv);
extern int (*p_rocsolver_dorgqr)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, const rocblas_int k,
                                 double *A, const rocblas_int lda,
                                 double *ipiv);
extern int (*p_rocsolver_cungqr)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, const rocblas_int k,
                                 rocblas_float_complex *A,
                                 const rocblas_int lda,
                                 rocblas_float_complex *ipiv);
extern int (*p_rocsolver_zungqr)(rocsolver_handle handle, const rocblas_int m,
                                 const rocblas_int n, const rocblas_int k,
                                 rocblas_double_complex *A,
                                 const rocblas_int lda,
                                 rocblas_double_complex *ipiv);
extern int (*p_rocsolver_sgels)(rocsolver_handle handle,
                                rocblas_operation trans, const rocblas_int m,
                                const rocblas_int n, const rocblas_int nrhs,
                                float *A, const rocblas_int lda, float *B,
                                const rocblas_int ldb, rocblas_int *info);
extern int (*p_rocsolver_dgels)(rocsolver_handle handle,
                                rocblas_operation trans, const rocblas_int m,
                                const rocblas_int n, const rocblas_int nrhs,
                                double *A, const rocblas_int lda, double *B,
                                const rocblas_int ldb, rocblas_int *info);
extern int (*p_rocsolver_cgels)(rocsolver_handle handle,
                                rocblas_operation trans, const rocblas_int m,
                                const rocblas_int n, const rocblas_int nrhs,
                                rocblas_float_complex *A, const rocblas_int lda,
                                rocblas_float_complex *B, const rocblas_int ldb,
                                rocblas_int *info);
extern int (*p_rocsolver_zgels)(rocsolver_handle handle,
                                rocblas_operation trans, const rocblas_int m,
                                const rocblas_int n, const rocblas_int nrhs,
                                rocblas_double_complex *A,
                                const rocblas_int lda,
                                rocblas_double_complex *B,
                                const rocblas_int ldb, rocblas_int *info);

bool loadOpenBLASLapack();
bool loadCuSolver();
bool loadRocSolver();

} // namespace dyn_lapack
} // namespace clap
