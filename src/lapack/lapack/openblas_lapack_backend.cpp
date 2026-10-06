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

#include "clap/lapack/lapack/openblas_lapack_backend.hpp"
#include "clap/lapack_dyn.hpp"
#include "clap/lapack/lapack_macros.hpp"
#include <stdexcept>
#include <iostream>

namespace clap {

// ── Helper: convert CLAP enums to LAPACKE characters ─────────────────────
static int to_lapack_layout(Layout l) {
    return (l == Layout::ColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
}

static char to_lapack_uplo(Uplo u) {
    return (u == Uplo::Upper) ? 'U' : 'L';
}

static char to_lapack_trans(Transpose t) {
    switch (t) {
        case Transpose::NoTrans:   return 'N';
        case Transpose::Trans:     return 'T';
        case Transpose::ConjTrans: return 'C';
    }
    return 'N';
}

static char to_lapack_diag(Diag d) {
    return (d == Diag::Unit) ? 'U' : 'N';
}

static char to_lapack_job(Job j) {
    return static_cast<char>(j);
}

// ── Constructor ───────────────────────────────────────────────────────────
OpenBlasLapackBackend::OpenBlasLapackBackend() {
    std::cout << "Using OpenBLAS LAPACK backend\n";
    if (!dyn_lapack::loadOpenBLASLapack())
        throw std::runtime_error("OpenBLAS LAPACK load failed");
}

// ══════════════════════════════════════════════════════════════════════════
// LINEAR SYSTEM SOLVERS
// ══════════════════════════════════════════════════════════════════════════

void OpenBlasLapackBackend::sgetrf(Layout layout,
                                   lapack_int m, lapack_int n,
                                   float  *A, lapack_int lda,
                                   lapack_int *ipiv, lapack_int *info) {
    *info = clap_LAPACKE_sgetrf(to_lapack_layout(layout), m, n, A, lda, ipiv);
}

void OpenBlasLapackBackend::dgetrf(Layout layout,
                                   lapack_int m, lapack_int n,
                                   double *A, lapack_int lda,
                                   lapack_int *ipiv, lapack_int *info) {
    *info = clap_LAPACKE_dgetrf(to_lapack_layout(layout), m, n, A, lda, ipiv);
}

void OpenBlasLapackBackend::sgetrs(Layout layout, Transpose trans,
                                   lapack_int n, lapack_int nrhs,
                                   const float  *A, lapack_int lda,
                                   const lapack_int *ipiv,
                                   float  *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_sgetrs(to_lapack_layout(layout),
                                 to_lapack_trans(trans),
                                 n, nrhs, A, lda, ipiv, B, ldb);
}

void OpenBlasLapackBackend::dgetrs(Layout layout, Transpose trans,
                                   lapack_int n, lapack_int nrhs,
                                   const double *A, lapack_int lda,
                                   const lapack_int *ipiv,
                                   double *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_dgetrs(to_lapack_layout(layout),
                                 to_lapack_trans(trans),
                                 n, nrhs, A, lda, ipiv, B, ldb);
}

void OpenBlasLapackBackend::sgesv(lapack_int n, lapack_int nrhs,
                                  float  *A, lapack_int lda, lapack_int *ipiv,
                                  float  *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, B, ldb);
}

void OpenBlasLapackBackend::dgesv(lapack_int n, lapack_int nrhs,
                                  double *A, lapack_int lda, lapack_int *ipiv,
                                  double *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, B, ldb);
}

void OpenBlasLapackBackend::sgetri(lapack_int n,
                                   float  *A, lapack_int lda,
                                   const lapack_int *ipiv, lapack_int *info) {
    *info = clap_LAPACKE_sgetri(LAPACK_COL_MAJOR, n, A, lda, ipiv);
}

void OpenBlasLapackBackend::dgetri(lapack_int n,
                                   double *A, lapack_int lda,
                                   const lapack_int *ipiv, lapack_int *info) {
    *info = clap_LAPACKE_dgetri(LAPACK_COL_MAJOR, n, A, lda, ipiv);
}

// ══════════════════════════════════════════════════════════════════════════
// CHOLESKY
// ══════════════════════════════════════════════════════════════════════════

void OpenBlasLapackBackend::spotrf(Layout layout, Uplo uplo, lapack_int n,
                                   float  *A, lapack_int lda, lapack_int *info) {
    *info = clap_LAPACKE_spotrf(to_lapack_layout(layout),
                                 to_lapack_uplo(uplo), n, A, lda);
}

void OpenBlasLapackBackend::dpotrf(Layout layout, Uplo uplo, lapack_int n,
                                   double *A, lapack_int lda, lapack_int *info) {
    *info = clap_LAPACKE_dpotrf(to_lapack_layout(layout),
                                 to_lapack_uplo(uplo), n, A, lda);
}

void OpenBlasLapackBackend::spotri(Layout layout, Uplo uplo, lapack_int n,
                                   float *A, lapack_int lda, lapack_int *info) {
    *info = clap_LAPACKE_spotri(to_lapack_layout(layout),
                                to_lapack_uplo(uplo), n, A, lda);
}

void OpenBlasLapackBackend::dpotri(Layout layout, Uplo uplo, lapack_int n,
                                   double *A, lapack_int lda, lapack_int *info) {
    *info = clap_LAPACKE_dpotri(to_lapack_layout(layout),
                                to_lapack_uplo(uplo), n, A, lda);
}

void OpenBlasLapackBackend::spotrs(Layout layout, Uplo uplo,
                                   lapack_int n, lapack_int nrhs,
                                   const float  *A, lapack_int lda,
                                   float  *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_spotrs(to_lapack_layout(layout),
                                 to_lapack_uplo(uplo), n, nrhs, A, lda, B, ldb);
}

void OpenBlasLapackBackend::dpotrs(Layout layout, Uplo uplo,
                                   lapack_int n, lapack_int nrhs,
                                   const double *A, lapack_int lda,
                                   double *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_dpotrs(to_lapack_layout(layout),
                                 to_lapack_uplo(uplo), n, nrhs, A, lda, B, ldb);
}

void OpenBlasLapackBackend::sposv(Layout layout, Uplo uplo,
                                  lapack_int n, lapack_int nrhs,
                                  float  *A, lapack_int lda,
                                  float  *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_sposv(to_lapack_layout(layout),
                                to_lapack_uplo(uplo), n, nrhs, A, lda, B, ldb);
}

void OpenBlasLapackBackend::dposv(Layout layout, Uplo uplo,
                                  lapack_int n, lapack_int nrhs,
                                  double *A, lapack_int lda,
                                  double *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_dposv(to_lapack_layout(layout),
                                to_lapack_uplo(uplo), n, nrhs, A, lda, B, ldb);
}

// ══════════════════════════════════════════════════════════════════════════
// QR
// ══════════════════════════════════════════════════════════════════════════

void OpenBlasLapackBackend::sgeqrf(Layout layout, lapack_int m, lapack_int n,
                                   float  *A, lapack_int lda, float  *tau,
                                   lapack_int *info) {
    *info = clap_LAPACKE_sgeqrf(to_lapack_layout(layout), m, n, A, lda, tau);
}

void OpenBlasLapackBackend::dgeqrf(Layout layout, lapack_int m, lapack_int n,
                                   double *A, lapack_int lda, double *tau,
                                   lapack_int *info) {
    *info = clap_LAPACKE_dgeqrf(to_lapack_layout(layout), m, n, A, lda, tau);
}

void OpenBlasLapackBackend::sorgqr(Layout layout,
                                   lapack_int m, lapack_int n, lapack_int k,
                                   float  *A, lapack_int lda,
                                   const float  *tau, lapack_int *info) {
    *info = clap_LAPACKE_sorgqr(to_lapack_layout(layout),
                                 m, n, k, A, lda, tau);
}

void OpenBlasLapackBackend::dorgqr(Layout layout,
                                   lapack_int m, lapack_int n, lapack_int k,
                                   double *A, lapack_int lda,
                                   const double *tau, lapack_int *info) {
    *info = clap_LAPACKE_dorgqr(to_lapack_layout(layout),
                                 m, n, k, A, lda, tau);
}

void OpenBlasLapackBackend::sgels(Layout layout, Transpose trans,
                                  lapack_int m, lapack_int n, lapack_int nrhs,
                                  float  *A, lapack_int lda,
                                  float  *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_sgels(to_lapack_layout(layout),
                                to_lapack_trans(trans),
                                m, n, nrhs, A, lda, B, ldb);
}

void OpenBlasLapackBackend::dgels(Layout layout, Transpose trans,
                                  lapack_int m, lapack_int n, lapack_int nrhs,
                                  double *A, lapack_int lda,
                                  double *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_dgels(to_lapack_layout(layout),
                                to_lapack_trans(trans),
                                m, n, nrhs, A, lda, B, ldb);
}

// ══════════════════════════════════════════════════════════════════════════
// EIGENVALUES
// ══════════════════════════════════════════════════════════════════════════

void OpenBlasLapackBackend::ssyev(Layout layout, Job jobz, Uplo uplo,
                                  lapack_int n,
                                  float  *A, lapack_int lda, float  *w,
                                  lapack_int *info) {
    *info = clap_LAPACKE_ssyev(to_lapack_layout(layout),
                                to_lapack_job(jobz),
                                to_lapack_uplo(uplo),
                                n, A, lda, w);
}

void OpenBlasLapackBackend::dsyev(Layout layout, Job jobz, Uplo uplo,
                                  lapack_int n,
                                  double *A, lapack_int lda, double *w,
                                  lapack_int *info) {
    *info = clap_LAPACKE_dsyev(to_lapack_layout(layout),
                                to_lapack_job(jobz),
                                to_lapack_uplo(uplo),
                                n, A, lda, w);
}

void OpenBlasLapackBackend::sgeev(Layout layout, Job jobvl, Job jobvr,
                                  lapack_int n,
                                  float  *A,  lapack_int lda,
                                  float  *wr, float  *wi,
                                  float  *VL, lapack_int ldvl,
                                  float  *VR, lapack_int ldvr,
                                  lapack_int *info) {
    *info = clap_LAPACKE_sgeev(to_lapack_layout(layout),
                                to_lapack_job(jobvl),
                                to_lapack_job(jobvr),
                                n, A, lda, wr, wi,
                                VL, ldvl, VR, ldvr);
}

void OpenBlasLapackBackend::dgeev(Layout layout, Job jobvl, Job jobvr,
                                  lapack_int n,
                                  double *A,  lapack_int lda,
                                  double *wr, double *wi,
                                  double *VL, lapack_int ldvl,
                                  double *VR, lapack_int ldvr,
                                  lapack_int *info) {
    *info = clap_LAPACKE_dgeev(to_lapack_layout(layout),
                                to_lapack_job(jobvl),
                                to_lapack_job(jobvr),
                                n, A, lda, wr, wi,
                                VL, ldvl, VR, ldvr);
}

// ══════════════════════════════════════════════════════════════════════════
// SVD
// ══════════════════════════════════════════════════════════════════════════

void OpenBlasLapackBackend::sgesvd(Layout layout, Job jobu, Job jobvt,
                                   lapack_int m, lapack_int n,
                                   float  *A, lapack_int lda, float  *s,
                                   float  *U, lapack_int ldu,
                                   float  *VT, lapack_int ldvt,
                                   float  *superb, lapack_int *info) {
    *info = clap_LAPACKE_sgesvd(to_lapack_layout(layout),
                                 to_lapack_job(jobu),
                                 to_lapack_job(jobvt),
                                 m, n, A, lda, s,
                                 U, ldu, VT, ldvt, superb);
}

void OpenBlasLapackBackend::dgesvd(Layout layout, Job jobu, Job jobvt,
                                   lapack_int m, lapack_int n,
                                   double *A, lapack_int lda, double *s,
                                   double *U, lapack_int ldu,
                                   double *VT, lapack_int ldvt,
                                   double *superb, lapack_int *info) {
    *info = clap_LAPACKE_dgesvd(to_lapack_layout(layout),
                                 to_lapack_job(jobu),
                                 to_lapack_job(jobvt),
                                 m, n, A, lda, s,
                                 U, ldu, VT, ldvt, superb);
}

// ══════════════════════════════════════════════════════════════════════════
// TRIANGULAR
// ══════════════════════════════════════════════════════════════════════════

void OpenBlasLapackBackend::strtrs(Layout layout, Uplo uplo,
                                   Transpose trans, Diag diag,
                                   lapack_int n, lapack_int nrhs,
                                   const float  *A, lapack_int lda,
                                   float  *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_strtrs(to_lapack_layout(layout),
                                 to_lapack_uplo(uplo),
                                 to_lapack_trans(trans),
                                 to_lapack_diag(diag),
                                 n, nrhs, A, lda, B, ldb);
}

void OpenBlasLapackBackend::dtrtrs(Layout layout, Uplo uplo,
                                   Transpose trans, Diag diag,
                                   lapack_int n, lapack_int nrhs,
                                   const double *A, lapack_int lda,
                                   double *B, lapack_int ldb, lapack_int *info) {
    *info = clap_LAPACKE_dtrtrs(to_lapack_layout(layout),
                                 to_lapack_uplo(uplo),
                                 to_lapack_trans(trans),
                                 to_lapack_diag(diag),
                                 n, nrhs, A, lda, B, ldb);
}

} // namespace clap
