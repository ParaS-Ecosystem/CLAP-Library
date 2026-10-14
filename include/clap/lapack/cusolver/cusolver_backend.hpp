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

class CuSolverBackend : public ILapackBackend {
public:
    CuSolverBackend();
    ~CuSolverBackend() override;

    void sgetrf(Layout layout, int m, int n,
                float  *A, int lda, int *ipiv,
                int *info) override;
    void dgetrf(Layout layout, int m, int n,
                double *A, int lda, int *ipiv,
                int *info) override;
    void cgetrf(Layout layout, int m, int n,
                cuComplex *A, int lda, int *ipiv,
                int *info) override;
    void zgetrf(Layout layout, int m, int n,
                cuDoubleComplex *A, int lda, int *ipiv,
                int *info) override;

    void sgetrs(Layout layout, Transpose trans,
                int n, int nrhs,
                const float  *A, int lda, const int *ipiv,
                float  *B, int ldb, int *info) override;
    void dgetrs(Layout layout, Transpose trans,
                int n, int nrhs,
                const double *A, int lda, const int *ipiv,
                double *B, int ldb, int *info) override;
    void cgetrs(Layout layout, Transpose trans,
                int n, int nrhs,
                const cuComplex *A, int lda, const int *ipiv,
                cuComplex *B, int ldb, int *info) override;
    void zgetrs(Layout layout, Transpose trans,
                int n, int nrhs,
                const cuDoubleComplex *A, int lda, const int *ipiv,
                cuDoubleComplex *B, int ldb, int *info) override;

    void sgesv(int n, int nrhs,
               float  *A, int lda, int *ipiv,
               float  *B, int ldb, int *info) override;
    void dgesv(int n, int nrhs,
               double *A, int lda, int *ipiv,
               double *B, int ldb, int *info) override;
    void cgesv(int n, int nrhs,
               cuComplex *A, int lda, int *ipiv,
               cuComplex *B, int ldb, int *info) override;
    void zgesv(int n, int nrhs,
               cuDoubleComplex *A, int lda, int *ipiv,
               cuDoubleComplex *B, int ldb, int *info) override;

    void sgetri(int n,
                float  *A, int lda, const int *ipiv,
                int *info) override;
    void dgetri(int n,
                double *A, int lda, const int *ipiv,
                int *info) override;
    void cgetri(int n,
                cuComplex *A, int lda, const int *ipiv,
                int *info) override;
    void zgetri(int n,
                cuDoubleComplex *A, int lda, const int *ipiv,
                int *info) override;

    void spotrf(Layout layout, Uplo uplo, int n,
                float  *A, int lda, int *info) override;
    void dpotrf(Layout layout, Uplo uplo, int n,
                double *A, int lda, int *info) override;
    void cpotrf(Layout layout, Uplo uplo, int n,
                cuComplex *A, int lda, int *info) override;
    void zpotrf(Layout layout, Uplo uplo, int n,
                cuDoubleComplex *A, int lda, int *info) override;

    void spotri(Layout layout, Uplo uplo, int n,
                float *A, int lda, int *info) override;

    void dpotri(Layout layout, Uplo uplo, int n,
                double *A, int lda, int *info) override;

    void spotrs(Layout layout, Uplo uplo, int n, int nrhs,
                const float  *A, int lda,
                float  *B, int ldb, int *info) override;
    void dpotrs(Layout layout, Uplo uplo, int n, int nrhs,
                const double *A, int lda,
                double *B, int ldb, int *info) override;

    void sposv(Layout layout, Uplo uplo, int n, int nrhs,
               float  *A, int lda,
               float  *B, int ldb, int *info) override;
    void dposv(Layout layout, Uplo uplo, int n, int nrhs,
               double *A, int lda,
               double *B, int ldb, int *info) override;

    void sgeqrf(Layout layout, int m, int n,
                float  *A, int lda, float  *tau,
                int *info) override;
    void dgeqrf(Layout layout, int m, int n,
                double *A, int lda, double *tau,
                int *info) override;

    void sorgqr(Layout layout, int m, int n, int k,
                float  *A, int lda, const float  *tau,
                int *info) override;
    void dorgqr(Layout layout, int m, int n, int k,
                double *A, int lda, const double *tau,
                int *info) override;

    void sgels(Layout layout, Transpose trans,
               int m, int n, int nrhs,
               float  *A, int lda,
               float  *B, int ldb, int *info) override;
    void dgels(Layout layout, Transpose trans,
               int m, int n, int nrhs,
               double *A, int lda,
               double *B, int ldb, int *info) override;

    void ssyev(Layout layout, Job jobz, Uplo uplo, int n,
               float  *A, int lda, float  *w,
               int *info) override;
    void dsyev(Layout layout, Job jobz, Uplo uplo, int n,
               double *A, int lda, double *w,
               int *info) override;

    void sgeev(Layout layout, Job jobvl, Job jobvr, int n,
               float  *A, int lda,
               float  *wr, float  *wi,
               float  *VL, int ldvl,
               float  *VR, int ldvr,
               int *info) override;
    void dgeev(Layout layout, Job jobvl, Job jobvr, int n,
               double *A, int lda,
               double *wr, double *wi,
               double *VL, int ldvl,
               double *VR, int ldvr,
               int *info) override;

    void sgesvd(Layout layout, Job jobu, Job jobvt,
                int m, int n,
                float  *A, int lda, float  *s,
                float  *U, int ldu,
                float  *VT, int ldvt,
                float  *superb, int *info) override;
    void dgesvd(Layout layout, Job jobu, Job jobvt,
                int m, int n,
                double *A, int lda, double *s,
                double *U, int ldu,
                double *VT, int ldvt,
                double *superb, int *info) override;

    void strtrs(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                int n, int nrhs,
                const float  *A, int lda,
                float  *B, int ldb, int *info) override;
    void dtrtrs(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                int n, int nrhs,
                const double *A, int lda,
                double *B, int ldb, int *info) override;

private:
    cusolverDnHandle_t m_handle;

    template<typename T>
    void getrf_impl(int m, int n,
                    T *A, int lda,
                    int *ipiv, int *info);

    template<typename T>
    void getrs_impl(Transpose trans, int n, int nrhs,
                    const T *A, int lda,
                    const int *ipiv,
                    T *B, int ldb, int *info);
    
    template<typename T>
    void gesv_impl(Transpose trans, int n, int nrhs,
                   T *A, int lda, int *ipiv,
                   T *B, int ldb, int *info);

    template<typename T>
    void potrf_impl(Uplo uplo, int n,
                    T *A, int lda, int *info);

    template<typename T>
    void potri_impl(Uplo uplo, int n, T *A, int lda,
                    int *info);

	template<typename T>
	void potrs_impl(Uplo uplo, int n, int nrhs,
               		 const T *A, int lda,
                	T *B, int ldb, int *info);

    template<typename T>
    void gesvd_impl(Job jobu, Job jobvt, int m, int n,
                    T *A, int lda, T *s,
                    T *U, int ldu,
                    T *VT, int ldvt,
                    int *info);

    template<typename T>
    void syev_impl(Job jobz, Uplo uplo, int n,
                   T *A, int lda, T *w, int *info);
};

}
