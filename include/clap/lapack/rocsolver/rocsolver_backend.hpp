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
#include "clap/lapack_factory.hpp"
#include "clap/lapack/lapack_types.hpp"

namespace clap {

class RocSolverBackend : public ILapackBackend {
public:
    RocSolverBackend();
    ~RocSolverBackend() override;

    // Linear solvers
    void sgetrf(Layout layout, lapack_int m, lapack_int n,
                float  *A, lapack_int lda, lapack_int *ipiv,
                lapack_int *info) override;
    void dgetrf(Layout layout, lapack_int m, lapack_int n,
                double *A, lapack_int lda, lapack_int *ipiv,
                lapack_int *info) override;

    void sgetrs(Layout layout, Transpose trans,
                lapack_int n, lapack_int nrhs,
                const float  *A, lapack_int lda, const lapack_int *ipiv,
                float  *B, lapack_int ldb, lapack_int *info) override;
    void dgetrs(Layout layout, Transpose trans,
                lapack_int n, lapack_int nrhs,
                const double *A, lapack_int lda, const lapack_int *ipiv,
                double *B, lapack_int ldb, lapack_int *info) override;

    void sgesv(lapack_int n, lapack_int nrhs,
               float  *A, lapack_int lda, lapack_int *ipiv,
               float  *B, lapack_int ldb, lapack_int *info) override;
    void dgesv(lapack_int n, lapack_int nrhs,
               double *A, lapack_int lda, lapack_int *ipiv,
               double *B, lapack_int ldb, lapack_int *info) override;

    void sgetri(lapack_int n,
                float  *A, lapack_int lda, const lapack_int *ipiv,
                lapack_int *info) override;
    void dgetri(lapack_int n,
                double *A, lapack_int lda, const lapack_int *ipiv,
                lapack_int *info) override;

    // Cholesky
    void spotrf(Layout layout, Uplo uplo, lapack_int n,
                float  *A, lapack_int lda, lapack_int *info) override;
    void dpotrf(Layout layout, Uplo uplo, lapack_int n,
                double *A, lapack_int lda, lapack_int *info) override;

    void spotri(Layout layout, Uplo uplo, lapack_int n,
                float *A, lapack_int lda, lapack_int *info) override;

    void dpotri(Layout layout, Uplo uplo, lapack_int n,
                double *A, lapack_int lda, lapack_int *info) override;

    void spotrs(Layout layout, Uplo uplo, lapack_int n, lapack_int nrhs,
                const float  *A, lapack_int lda,
                float  *B, lapack_int ldb, lapack_int *info) override;
    void dpotrs(Layout layout, Uplo uplo, lapack_int n, lapack_int nrhs,
                const double *A, lapack_int lda,
                double *B, lapack_int ldb, lapack_int *info) override;

    void sposv(Layout layout, Uplo uplo, lapack_int n, lapack_int nrhs,
               float  *A, lapack_int lda,
               float  *B, lapack_int ldb, lapack_int *info) override;
    void dposv(Layout layout, Uplo uplo, lapack_int n, lapack_int nrhs,
               double *A, lapack_int lda,
               double *B, lapack_int ldb, lapack_int *info) override;

    // QR
    void sgeqrf(Layout layout, lapack_int m, lapack_int n,
                float  *A, lapack_int lda, float  *tau,
                lapack_int *info) override;
    void dgeqrf(Layout layout, lapack_int m, lapack_int n,
                double *A, lapack_int lda, double *tau,
                lapack_int *info) override;

    void sorgqr(Layout layout, lapack_int m, lapack_int n, lapack_int k,
                float  *A, lapack_int lda, const float  *tau,
                lapack_int *info) override;
    void dorgqr(Layout layout, lapack_int m, lapack_int n, lapack_int k,
                double *A, lapack_int lda, const double *tau,
                lapack_int *info) override;

    void sgels(Layout layout, Transpose trans,
               lapack_int m, lapack_int n, lapack_int nrhs,
               float  *A, lapack_int lda,
               float  *B, lapack_int ldb, lapack_int *info) override;
    void dgels(Layout layout, Transpose trans,
               lapack_int m, lapack_int n, lapack_int nrhs,
               double *A, lapack_int lda,
               double *B, lapack_int ldb, lapack_int *info) override;

    // Eigenvalues
    void ssyev(Layout layout, Job jobz, Uplo uplo, lapack_int n,
               float  *A, lapack_int lda, float  *w,
               lapack_int *info) override;
    void dsyev(Layout layout, Job jobz, Uplo uplo, lapack_int n,
               double *A, lapack_int lda, double *w,
               lapack_int *info) override;

    void sgeev(Layout layout, Job jobvl, Job jobvr, lapack_int n,
               float  *A, lapack_int lda,
               float  *wr, float  *wi,
               float  *VL, lapack_int ldvl,
               float  *VR, lapack_int ldvr,
               lapack_int *info) override;
    void dgeev(Layout layout, Job jobvl, Job jobvr, lapack_int n,
               double *A, lapack_int lda,
               double *wr, double *wi,
               double *VL, lapack_int ldvl,
               double *VR, lapack_int ldvr,
               lapack_int *info) override;

    // SVD
    void sgesvd(Layout layout, Job jobu, Job jobvt,
                lapack_int m, lapack_int n,
                float  *A, lapack_int lda, float  *s,
                float  *U, lapack_int ldu,
                float  *VT, lapack_int ldvt,
                float  *superb, lapack_int *info) override;
    void dgesvd(Layout layout, Job jobu, Job jobvt,
                lapack_int m, lapack_int n,
                double *A, lapack_int lda, double *s,
                double *U, lapack_int ldu,
                double *VT, lapack_int ldvt,
                double *superb, lapack_int *info) override;

    // Triangular
    void strtrs(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                lapack_int n, lapack_int nrhs,
                const float  *A, lapack_int lda,
                float  *B, lapack_int ldb, lapack_int *info) override;
    void dtrtrs(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                lapack_int n, lapack_int nrhs,
                const double *A, lapack_int lda,
                double *B, lapack_int ldb, lapack_int *info) override;

private:
    rocsolver_handle m_handle;

    template<typename T>
    void getrf_impl(lapack_int m, lapack_int n,
                    T *A, lapack_int lda,
                    lapack_int *ipiv, lapack_int *info);

    template<typename T>
    void getrs_impl(Transpose trans, lapack_int n, lapack_int nrhs,
                    const T *A, lapack_int lda,
                    const lapack_int *ipiv,
                    T *B, lapack_int ldb, lapack_int *info);

    template<typename T>
    void potrf_impl(Uplo uplo, lapack_int n,
                    T *A, lapack_int lda, lapack_int *info);

    template<typename T>
    void potri_impl(Uplo uplo, lapack_int n,
                    T *A, lapack_int lda, lapack_int *info);
    
	template<typename T>
	void potrs_impl(Uplo uplo, lapack_int n, lapack_int nrhs,
               		const T *A, lapack_int lda,
                	T *B, lapack_int ldb,
               		lapack_int *info);

	template<typename T>
    void gesvd_impl(Job jobu, Job jobvt, lapack_int m, lapack_int n,
                    T *A, lapack_int lda, T *s,
                    T *U, lapack_int ldu,
                    T *VT, lapack_int ldvt, lapack_int *info);

    template<typename T>
    void syev_impl(Job jobz, Uplo uplo, lapack_int n,
                   T *A, lapack_int lda, T *w, lapack_int *info);
};

} // namespace clap
