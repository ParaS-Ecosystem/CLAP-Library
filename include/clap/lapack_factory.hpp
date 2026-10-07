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
#include "clap/blas_factory.hpp"
#include "clap/lapack/lapack_types.hpp"
#include <memory>

namespace clap {

// ── ILapackBackend ────────────────────────────────────────────────────────
// Unified LAPACK interface following the same pattern as IBlasBackend.
// All routines take CLAP enum types and host pointers.
// The backend handles all device memory, transfers, and synchronization.

class ILapackBackend {
public:
    virtual ~ILapackBackend() = default;

    // ══════════════════════════════════════════════════════════════════════
    // LINEAR SYSTEM SOLVERS
    // ══════════════════════════════════════════════════════════════════════

    // ── getrf: LU factorization with partial pivoting ─────────────────────
    // A = P * L * U
    // A (in)  : m×n matrix
    // A (out) : L and U factors (L has implicit unit diagonal)
    // ipiv    : pivot indices, length min(m,n)
    // info    : 0 = success, >0 = singular at row info
    virtual void sgetrf(Layout layout,
                        lapack_int m, lapack_int n,
                        float  *A, lapack_int lda,
                        lapack_int *ipiv,
                        lapack_int *info) = 0;

    virtual void dgetrf(Layout layout,
                        lapack_int m, lapack_int n,
                        double *A, lapack_int lda,
                        lapack_int *ipiv,
                        lapack_int *info) = 0;

    // ── getrs: solve AX = B using LU factorization from getrf ────────────
    // A    : factored matrix from getrf (not modified)
    // ipiv : pivot indices from getrf
    // B    : right-hand side (in), solution (out)
    virtual void sgetrs(Layout layout, Transpose trans,
                        lapack_int n, lapack_int nrhs,
                        const float  *A, lapack_int lda,
                        const lapack_int *ipiv,
                        float  *B, lapack_int ldb,
                        lapack_int *info) = 0;

    virtual void dgetrs(Layout layout, Transpose trans,
                        lapack_int n, lapack_int nrhs,
                        const double *A, lapack_int lda,
                        const lapack_int *ipiv,
                        double *B, lapack_int ldb,
                        lapack_int *info) = 0;

    // ── gesv: solve AX = B (driver: getrf + getrs in one call) ───────────
    // A    : n×n matrix (overwritten with LU)
    // ipiv : pivot indices (output)
    // B    : right-hand side (in), solution (out)
    virtual void sgesv(lapack_int n, lapack_int nrhs,
                       float  *A, lapack_int lda,
                       lapack_int *ipiv,
                       float  *B, lapack_int ldb,
                       lapack_int *info) = 0;

    virtual void dgesv(lapack_int n, lapack_int nrhs,
                       double *A, lapack_int lda,
                       lapack_int *ipiv,
                       double *B, lapack_int ldb,
                       lapack_int *info) = 0;

    // ── getri: compute matrix inverse using LU factorization ─────────────
    // A    : factored matrix from getrf (overwritten with inverse)
    // ipiv : pivot indices from getrf
    virtual void sgetri(lapack_int n,
                        float  *A, lapack_int lda,
                        const lapack_int *ipiv,
                        lapack_int *info) = 0;

    virtual void dgetri(lapack_int n,
                        double *A, lapack_int lda,
                        const lapack_int *ipiv,
                        lapack_int *info) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // CHOLESKY ROUTINES
    // ══════════════════════════════════════════════════════════════════════

    // ── potrf: Cholesky factorization A = U^T*U or A = L*L^T ─────────────
    // uplo : Upper or Lower triangle stored
    // A    : symmetric positive definite n×n matrix (overwritten with factor)
    virtual void spotrf(Layout layout, Uplo uplo,
                        lapack_int n,
                        float  *A, lapack_int lda,
                        lapack_int *info) = 0;

    virtual void dpotrf(Layout layout, Uplo uplo,
                        lapack_int n,
                        double *A, lapack_int lda,
                        lapack_int *info) = 0;
                        
    // ── potri: compute inverse from Cholesky factor ───────────────────────
    virtual void spotri(Layout layout, Uplo uplo,
                        lapack_int n,
                        float *A, lapack_int lda,
                        lapack_int *info) = 0;

    virtual void dpotri(Layout layout, Uplo uplo,
                        lapack_int n,
                        double *A, lapack_int lda,
                        lapack_int *info) = 0;
    
    // ── potrs: solve AX = B using Cholesky factor from potrf ─────────────
    virtual void spotrs(Layout layout, Uplo uplo,
                        lapack_int n, lapack_int nrhs,
                        const float  *A, lapack_int lda,
                        float  *B, lapack_int ldb,
                        lapack_int *info) = 0;

    virtual void dpotrs(Layout layout, Uplo uplo,
                        lapack_int n, lapack_int nrhs,
                        const double *A, lapack_int lda,
                        double *B, lapack_int ldb,
                        lapack_int *info) = 0;

    // ── posv: symmetric positive definite solve (driver) ─────────────────
    virtual void sposv(Layout layout, Uplo uplo,
                       lapack_int n, lapack_int nrhs,
                       float  *A, lapack_int lda,
                       float  *B, lapack_int ldb,
                       lapack_int *info) = 0;

    virtual void dposv(Layout layout, Uplo uplo,
                       lapack_int n, lapack_int nrhs,
                       double *A, lapack_int lda,
                       double *B, lapack_int ldb,
                       lapack_int *info) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // QR FACTORIZATION AND LEAST SQUARES
    // ══════════════════════════════════════════════════════════════════════

    // ── geqrf: QR factorization A = Q * R ────────────────────────────────
    // A    : m×n matrix (overwritten: upper triangle = R, lower = Householder)
    // tau  : Householder scalar factors, length min(m,n)
    virtual void sgeqrf(Layout layout,
                        lapack_int m, lapack_int n,
                        float  *A, lapack_int lda,
                        float  *tau,
                        lapack_int *info) = 0;

    virtual void dgeqrf(Layout layout,
                        lapack_int m, lapack_int n,
                        double *A, lapack_int lda,
                        double *tau,
                        lapack_int *info) = 0;

    // ── orgqr: generate Q from QR factorization (float) ──────────────────
    // dormqr (double) / sormqr (float): apply Q to matrix C
    virtual void sorgqr(Layout layout,
                        lapack_int m, lapack_int n, lapack_int k,
                        float  *A, lapack_int lda,
                        const float  *tau,
                        lapack_int *info) = 0;

    virtual void dorgqr(Layout layout,
                        lapack_int m, lapack_int n, lapack_int k,
                        double *A, lapack_int lda,
                        const double *tau,
                        lapack_int *info) = 0;

    // ── gels: least squares min||AX-B||_2 using QR or LQ ─────────────────
    // trans: NoTrans solves min||AX-B||, Trans solves min||A^TX-B||
    // A    : m×n matrix (overwritten)
    // B    : m×nrhs RHS (in), solution (out, size max(m,n)×nrhs)
    virtual void sgels(Layout layout, Transpose trans,
                       lapack_int m, lapack_int n, lapack_int nrhs,
                       float  *A, lapack_int lda,
                       float  *B, lapack_int ldb,
                       lapack_int *info) = 0;

    virtual void dgels(Layout layout, Transpose trans,
                       lapack_int m, lapack_int n, lapack_int nrhs,
                       double *A, lapack_int lda,
                       double *B, lapack_int ldb,
                       lapack_int *info) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // EIGENVALUE AND SINGULAR VALUE ROUTINES
    // ══════════════════════════════════════════════════════════════════════

    // ── syev: eigenvalues and eigenvectors of symmetric matrix ────────────
    // jobz : Job::NoVec = eigenvalues only, Job::Vec = also eigenvectors
    // uplo : Upper or Lower triangle stored
    // A    : symmetric n×n matrix; overwritten with eigenvectors if jobz=Vec
    // w    : eigenvalues in ascending order (output)
    virtual void ssyev(Layout layout, Job jobz, Uplo uplo,
                       lapack_int n,
                       float  *A, lapack_int lda,
                       float  *w,
                       lapack_int *info) = 0;

    virtual void dsyev(Layout layout, Job jobz, Uplo uplo,
                       lapack_int n,
                       double *A, lapack_int lda,
                       double *w,
                       lapack_int *info) = 0;

    // ── geev: eigenvalues of general non-symmetric matrix ─────────────────
    // jobvl : Job::NoVec or Job::Vec for left eigenvectors
    // jobvr : Job::NoVec or Job::Vec for right eigenvectors
    // wr, wi: real and imaginary parts of eigenvalues
    // VL    : left eigenvectors (n×n)
    // VR    : right eigenvectors (n×n)
    virtual void sgeev(Layout layout, Job jobvl, Job jobvr,
                       lapack_int n,
                       float  *A,  lapack_int lda,
                       float  *wr, float  *wi,
                       float  *VL, lapack_int ldvl,
                       float  *VR, lapack_int ldvr,
                       lapack_int *info) = 0;

    virtual void dgeev(Layout layout, Job jobvl, Job jobvr,
                       lapack_int n,
                       double *A,  lapack_int lda,
                       double *wr, double *wi,
                       double *VL, lapack_int ldvl,
                       double *VR, lapack_int ldvr,
                       lapack_int *info) = 0;

    // ── gesvd: singular value decomposition A = U * S * V^T ──────────────
    // jobu  : Job for left singular vectors U
    // jobvt : Job for right singular vectors V^T
    // s     : singular values in descending order (output)
    // U     : left singular vectors (m×m or m×min(m,n))
    // VT    : right singular vectors transposed (n×n or min(m,n)×n)
    // superb: superdiagonal of bidiagonal (used internally)
    virtual void sgesvd(Layout layout, Job jobu, Job jobvt,
                        lapack_int m, lapack_int n,
                        float  *A, lapack_int lda,
                        float  *s,
                        float  *U,  lapack_int ldu,
                        float  *VT, lapack_int ldvt,
                        float  *superb,
                        lapack_int *info) = 0;

    virtual void dgesvd(Layout layout, Job jobu, Job jobvt,
                        lapack_int m, lapack_int n,
                        double *A, lapack_int lda,
                        double *s,
                        double *U,  lapack_int ldu,
                        double *VT, lapack_int ldvt,
                        double *superb,
                        lapack_int *info) = 0;

    // ══════════════════════════════════════════════════════════════════════
    // TRIANGULAR ROUTINES
    // ══════════════════════════════════════════════════════════════════════

    // ── trtrs: solve triangular system A*X = B or A^T*X = B ──────────────
    virtual void strtrs(Layout layout, Uplo uplo,
                        Transpose trans, Diag diag,
                        lapack_int n, lapack_int nrhs,
                        const float  *A, lapack_int lda,
                        float  *B, lapack_int ldb,
                        lapack_int *info) = 0;

    virtual void dtrtrs(Layout layout, Uplo uplo,
                        Transpose trans, Diag diag,
                        lapack_int n, lapack_int nrhs,
                        const double *A, lapack_int lda,
                        double *B, lapack_int ldb,
                        lapack_int *info) = 0;
};

// ── LapackFactory ─────────────────────────────────────────────────────────
// Mirrors BlasFactory — creates backend based on same BackendType enum.
// CPU backend uses OpenBLAS/LAPACK via dlopen.
// CUDA backend uses cuSOLVER via dlopen.
// ROCM backend uses rocSOLVER via dlopen.

class LapackFactory {
public:
    static std::unique_ptr<ILapackBackend>
    create(BackendType default_backend = BackendType::CPU);
};

} // namespace clap
