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

// Dynamic loader declarations for LAPACK (OpenBLAS), cuSOLVER, rocSOLVER.
// Follows the exact same pattern as dyn_backends.hpp for BLAS.

#pragma once
#include "clap/lapack/lapack_types.hpp"
#include <cstdint>
#include <dlfcn.h>
#include <iostream>

namespace clap {
namespace dyn_lapack {

// ══════════════════════════════════════════════════════════════════════════
// OpenBLAS LAPACK function pointers (Fortran-style via LAPACKE C interface)
// ══════════════════════════════════════════════════════════════════════════

// ── Linear solvers ────────────────────────────────────────────────────────
extern lapack_int (*p_LAPACKE_sgetrf)(int matrix_layout,
                                      lapack_int m, lapack_int n,
                                      float  *a, lapack_int lda,
                                      lapack_int *ipiv);
extern lapack_int (*p_LAPACKE_dgetrf)(int matrix_layout,
                                      lapack_int m, lapack_int n,
                                      double *a, lapack_int lda,
                                      lapack_int *ipiv);

extern lapack_int (*p_LAPACKE_sgetrs)(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const float  *a, lapack_int lda,
                                      const lapack_int *ipiv,
                                      float  *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dgetrs)(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const double *a, lapack_int lda,
                                      const lapack_int *ipiv,
                                      double *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_sgesv)(int matrix_layout,
                                     lapack_int n, lapack_int nrhs,
                                     float  *a, lapack_int lda,
                                     lapack_int *ipiv,
                                     float  *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dgesv)(int matrix_layout,
                                     lapack_int n, lapack_int nrhs,
                                     double *a, lapack_int lda,
                                     lapack_int *ipiv,
                                     double *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_sgetri)(int matrix_layout,
                                      lapack_int n,
                                      float  *a, lapack_int lda,
                                      const lapack_int *ipiv);
extern lapack_int (*p_LAPACKE_dgetri)(int matrix_layout,
                                      lapack_int n,
                                      double *a, lapack_int lda,
                                      const lapack_int *ipiv);

// ── Cholesky ──────────────────────────────────────────────────────────────
extern lapack_int (*p_LAPACKE_spotrf)(int matrix_layout, char uplo,
                                      lapack_int n,
                                      float  *a, lapack_int lda);
extern lapack_int (*p_LAPACKE_dpotrf)(int matrix_layout, char uplo,
                                      lapack_int n,
                                      double *a, lapack_int lda);

extern lapack_int (*p_LAPACKE_spotri)(int matrix_layout, char uplo,
                                      lapack_int n, float *A, lapack_int lda);

extern lapack_int (*p_LAPACKE_dpotri)(int matrix_layout, char uplo,
                                      lapack_int n, double *A, lapack_int lda);
                                      
extern lapack_int (*p_LAPACKE_spotrs)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float  *a, lapack_int lda,
                                      float  *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dpotrs)(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double *a, lapack_int lda,
                                      double *b, lapack_int ldb);

extern lapack_int (*p_LAPACKE_sposv)(int matrix_layout, char uplo,
                                     lapack_int n, lapack_int nrhs,
                                     float  *a, lapack_int lda,
                                     float  *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dposv)(int matrix_layout, char uplo,
                                     lapack_int n, lapack_int nrhs,
                                     double *a, lapack_int lda,
                                     double *b, lapack_int ldb);

// ── QR factorization ──────────────────────────────────────────────────────
extern lapack_int (*p_LAPACKE_sgeqrf)(int matrix_layout,
                                      lapack_int m, lapack_int n,
                                      float  *a, lapack_int lda,
                                      float  *tau);
extern lapack_int (*p_LAPACKE_dgeqrf)(int matrix_layout,
                                      lapack_int m, lapack_int n,
                                      double *a, lapack_int lda,
                                      double *tau);

extern lapack_int (*p_LAPACKE_sorgqr)(int matrix_layout,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      float  *a, lapack_int lda,
                                      const float  *tau);
extern lapack_int (*p_LAPACKE_dorgqr)(int matrix_layout,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      double *a, lapack_int lda,
                                      const double *tau);

extern lapack_int (*p_LAPACKE_sgels)(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n, lapack_int nrhs,
                                     float  *a, lapack_int lda,
                                     float  *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dgels)(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n, lapack_int nrhs,
                                     double *a, lapack_int lda,
                                     double *b, lapack_int ldb);

// ── Eigenvalues ───────────────────────────────────────────────────────────
extern lapack_int (*p_LAPACKE_ssyev)(int matrix_layout, char jobz, char uplo,
                                     lapack_int n,
                                     float  *a, lapack_int lda,
                                     float  *w);
extern lapack_int (*p_LAPACKE_dsyev)(int matrix_layout, char jobz, char uplo,
                                     lapack_int n,
                                     double *a, lapack_int lda,
                                     double *w);

extern lapack_int (*p_LAPACKE_sgeev)(int matrix_layout,
                                     char jobvl, char jobvr,
                                     lapack_int n,
                                     float  *a,  lapack_int lda,
                                     float  *wr, float  *wi,
                                     float  *vl, lapack_int ldvl,
                                     float  *vr, lapack_int ldvr);
extern lapack_int (*p_LAPACKE_dgeev)(int matrix_layout,
                                     char jobvl, char jobvr,
                                     lapack_int n,
                                     double *a,  lapack_int lda,
                                     double *wr, double *wi,
                                     double *vl, lapack_int ldvl,
                                     double *vr, lapack_int ldvr);

// ── SVD ───────────────────────────────────────────────────────────────────
extern lapack_int (*p_LAPACKE_sgesvd)(int matrix_layout,
                                      char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      float  *a,  lapack_int lda,
                                      float  *s,
                                      float  *u,  lapack_int ldu,
                                      float  *vt, lapack_int ldvt,
                                      float  *superb);
extern lapack_int (*p_LAPACKE_dgesvd)(int matrix_layout,
                                      char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      double *a,  lapack_int lda,
                                      double *s,
                                      double *u,  lapack_int ldu,
                                      double *vt, lapack_int ldvt,
                                      double *superb);

// ── Triangular solve ──────────────────────────────────────────────────────
extern lapack_int (*p_LAPACKE_strtrs)(int matrix_layout,
                                      char uplo, char trans, char diag,
                                      lapack_int n, lapack_int nrhs,
                                      const float  *a, lapack_int lda,
                                      float  *b, lapack_int ldb);
extern lapack_int (*p_LAPACKE_dtrtrs)(int matrix_layout,
                                      char uplo, char trans, char diag,
                                      lapack_int n, lapack_int nrhs,
                                      const double *a, lapack_int lda,
                                      double *b, lapack_int ldb);

// ══════════════════════════════════════════════════════════════════════════
// cuSOLVER function pointers
// ══════════════════════════════════════════════════════════════════════════

extern cusolverStatus_t (*p_cusolverDnCreate)(cusolverDnHandle_t *handle);
extern cusolverStatus_t (*p_cusolverDnDestroy)(cusolverDnHandle_t handle);

// ── getrf buffer size + factorization ─────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSgetrf_bufferSize)(
    cusolverDnHandle_t handle,
    int m, int n,
    float *A, int lda,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDgetrf_bufferSize)(
    cusolverDnHandle_t handle,
    int m, int n,
    double *A, int lda,
    int *Lwork);

extern cusolverStatus_t (*p_cusolverDnSgetrf)(
    cusolverDnHandle_t handle,
    int m, int n,
    float *A, int lda,
    float *Workspace,
    int *devIpiv, int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgetrf)(
    cusolverDnHandle_t handle,
    int m, int n,
    double *A, int lda,
    double *Workspace,
    int *devIpiv, int *devInfo);

// ── getrs ─────────────────────────────────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSgetrs)(
    cusolverDnHandle_t handle,
    int trans,
    int n, int nrhs,
    const float *A, int lda,
    const int *devIpiv,
    float *B, int ldb,
    int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgetrs)(
    cusolverDnHandle_t handle,
    int trans,
    int n, int nrhs,
    const double *A, int lda,
    const int *devIpiv,
    double *B, int ldb,
    int *devInfo);

// ── potrf ─────────────────────────────────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSpotrf_bufferSize)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    float *A, int lda,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDpotrf_bufferSize)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    double *A, int lda,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnSpotrf)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    float *A, int lda,
    float *Workspace, int Lwork,
    int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDpotrf)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    double *A, int lda,
    double *Workspace, int Lwork,
    int *devInfo);

// ── potri ─────────────────────────────────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSpotri_bufferSize)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    float *A, int lda,
    int *Lwork );

extern cusolverStatus_t (*p_cusolverDnDpotri_bufferSize)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    double *A, int lda,
    int *Lwork );

extern cusolverStatus_t (*p_cusolverDnSpotri)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    float *A, int lda,
    float *Workspace, int Lwork,
    int *devInfo );

extern cusolverStatus_t (*p_cusolverDnDpotri)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    double *A, int lda,
    double *Workspace, int Lwork,
    int *devInfo );

// ── potrs ─────────────────────────────────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSpotrs)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    int nrhs, const float *A,
    int lda, float *B,
    int ldb, int *devInfo);

extern cusolverStatus_t (*p_cusolverDnDpotrs)(
    cusolverDnHandle_t handle,
    int uplo, int n,
    int nrhs, const double *A,
    int lda, double *B,
    int ldb, int *devInfo);

// ── gesvd ─────────────────────────────────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSgesvd_bufferSize)(
    cusolverDnHandle_t handle,
    int m, int n,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnDgesvd_bufferSize)(
    cusolverDnHandle_t handle,
    int m, int n,
    int *Lwork);
extern cusolverStatus_t (*p_cusolverDnSgesvd)(
    cusolverDnHandle_t handle,
    char jobu, char jobvt,
    int m, int n,
    float *A, int lda,
    float *S,
    float *U,  int ldu,
    float *VT, int ldvt,
    float *Work, int Lwork,
    float *rwork,
    int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDgesvd)(
    cusolverDnHandle_t handle,
    char jobu, char jobvt,
    int m, int n,
    double *A, int lda,
    double *S,
    double *U,  int ldu,
    double *VT, int ldvt,
    double *Work, int Lwork,
    double *rwork,
    int *devInfo);

// ── syev (dsyevd on GPU) ──────────────────────────────────────────────────
extern cusolverStatus_t (*p_cusolverDnSsyevd_bufferSize)(
    cusolverDnHandle_t handle,
    int jobz, int uplo,
    int n,
    const float *A, int lda,
    const float *W,
    int *lwork);
extern cusolverStatus_t (*p_cusolverDnDsyevd_bufferSize)(
    cusolverDnHandle_t handle,
    int jobz, int uplo,
    int n,
    const double *A, int lda,
    const double *W,
    int *lwork);
extern cusolverStatus_t (*p_cusolverDnSsyevd)(
    cusolverDnHandle_t handle,
    int jobz, int uplo,
    int n,
    float *A, int lda,
    float *W,
    float *work, int lwork,
    int *devInfo);
extern cusolverStatus_t (*p_cusolverDnDsyevd)(
    cusolverDnHandle_t handle,
    int jobz, int uplo,
    int n,
    double *A, int lda,
    double *W,
    double *work, int lwork,
    int *devInfo);

// ══════════════════════════════════════════════════════════════════════════
// rocSOLVER function pointers
// ══════════════════════════════════════════════════════════════════════════

extern int (*p_rocsolver_sgetrf)(rocsolver_handle handle,
                                  rocblas_int_solver m, rocblas_int_solver n,
                                  float  *A, rocblas_int_solver lda,
                                  rocblas_int_solver *ipiv,
                                  rocblas_int_solver *info);
extern int (*p_rocsolver_dgetrf)(rocsolver_handle handle,
                                  rocblas_int_solver m, rocblas_int_solver n,
                                  double *A, rocblas_int_solver lda,
                                  rocblas_int_solver *ipiv,
                                  rocblas_int_solver *info);

extern int (*p_rocsolver_sgetrs)(rocsolver_handle handle,
                                  int trans,
                                  rocblas_int_solver n, rocblas_int_solver nrhs,
                                  const float  *A, rocblas_int_solver lda,
                                  const rocblas_int_solver *ipiv,
                                  float  *B, rocblas_int_solver ldb);
extern int (*p_rocsolver_dgetrs)(rocsolver_handle handle,
                                  int trans,
                                  rocblas_int_solver n, rocblas_int_solver nrhs,
                                  const double *A, rocblas_int_solver lda,
                                  const rocblas_int_solver *ipiv,
                                  double *B, rocblas_int_solver ldb);

extern int (*p_rocsolver_spotrf)(rocsolver_handle handle,
                                  int uplo,
                                  rocblas_int_solver n,
                                  float  *A, rocblas_int_solver lda,
                                  rocblas_int_solver *info);
extern int (*p_rocsolver_dpotrf)(rocsolver_handle handle,
                                  int uplo,
                                  rocblas_int_solver n,
                                  double *A, rocblas_int_solver lda,
                                  rocblas_int_solver *info);

extern int (*p_rocsolver_spotri)(rocsolver_handle handle, 
                                int uplo, rocblas_int_solver n, 
                                float *A, rocblas_int_solver lda, 
                                rocblas_int_solver *info);

extern int (*p_rocsolver_dpotri)(rocsolver_handle handle, 
                                int uplo, rocblas_int_solver n, 
                                double *A, rocblas_int_solver lda, 
                                rocblas_int_solver *info);

extern int (*p_rocsolver_spotrs)(rocsolver_handle handle,
    							int uplo,
    							rocblas_int_solver n,
    							rocblas_int_solver nrhs,
    							const float *A,
   								rocblas_int_solver lda,
    							float *B,
    							rocblas_int_solver ldb);

extern int (*p_rocsolver_dpotrs)(rocsolver_handle handle,
    							int uplo,
    							rocblas_int_solver n,
    							rocblas_int_solver nrhs,
    							const double *A,
    							rocblas_int_solver lda,
    							double *B,
    							rocblas_int_solver ldb);

extern int (*p_rocsolver_sgesvd)(rocsolver_handle handle,
                                  int jobu, int jobv,
                                  rocblas_int_solver m, rocblas_int_solver n,
                                  float  *A, rocblas_int_solver lda,
                                  float  *S,
                                  float  *U,  rocblas_int_solver ldu,
                                  float  *V,  rocblas_int_solver ldv,
                                  float  *E,
                                  int fast_alg,
                                  rocblas_int_solver *info);
extern int (*p_rocsolver_dgesvd)(rocsolver_handle handle,
                                  int jobu, int jobv,
                                  rocblas_int_solver m, rocblas_int_solver n,
                                  double *A, rocblas_int_solver lda,
                                  double *S,
                                  double *U,  rocblas_int_solver ldu,
                                  double *V,  rocblas_int_solver ldv,
                                  double *E,
                                  int fast_alg,
                                  rocblas_int_solver *info);

extern int (*p_rocsolver_ssyev)(rocsolver_handle handle,
                                 int evect, int uplo,
                                 rocblas_int_solver n,
                                 float  *A, rocblas_int_solver lda,
                                 float  *D, float  *E,
                                 rocblas_int_solver *info);
extern int (*p_rocsolver_dsyev)(rocsolver_handle handle,
                                 int evect, int uplo,
                                 rocblas_int_solver n,
                                 double *A, rocblas_int_solver lda,
                                 double *D, double *E,
                                 rocblas_int_solver *info);

// ── Loader functions ──────────────────────────────────────────────────────
bool loadOpenBLASLapack();
bool loadCuSolver();
bool loadRocSolver();

} // namespace dyn_lapack
} // namespace clap
