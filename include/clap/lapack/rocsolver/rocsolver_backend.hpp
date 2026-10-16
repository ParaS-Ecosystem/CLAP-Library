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

    void sgetrf(Layout layout, rocblas_int m, rocblas_int n,
                float  *A, rocblas_int lda, rocblas_int *ipiv,
                rocblas_int *info) override;
    void dgetrf(Layout layout, rocblas_int m, rocblas_int n,
                double *A, rocblas_int lda, rocblas_int *ipiv,
                rocblas_int *info) override;
    void cgetrf(Layout layout, rocblas_int m, rocblas_int n,
                rocblas_float_complex *A, rocblas_int lda, rocblas_int *ipiv,
                rocblas_int *info) override;
    void zgetrf(Layout layout, rocblas_int m, rocblas_int n,
                rocblas_double_complex *A, rocblas_int lda, rocblas_int *ipiv,
                rocblas_int *info) override;

    void sgetrs(Layout layout, Transpose trans,
                rocblas_int n, rocblas_int nrhs,
                const float  *A, rocblas_int lda, const rocblas_int *ipiv,
                float  *B, rocblas_int ldb, rocblas_int *info) override;
    void dgetrs(Layout layout, Transpose trans,
                rocblas_int n, rocblas_int nrhs,
                const double *A, rocblas_int lda, const rocblas_int *ipiv,
                double *B, rocblas_int ldb, rocblas_int *info) override;
    void cgetrs(Layout layout, Transpose trans,
                rocblas_int n, rocblas_int nrhs,
                const rocblas_float_complex *A, rocblas_int lda, const rocblas_int *ipiv,
                rocblas_float_complex *B, rocblas_int ldb, rocblas_int *info) override;
    void zgetrs(Layout layout, Transpose trans,
                rocblas_int n, rocblas_int nrhs,
                const rocblas_double_complex *A, rocblas_int lda, const rocblas_int *ipiv,
                rocblas_double_complex *B, rocblas_int ldb, rocblas_int *info) override;

    void sgesv(rocblas_int n, rocblas_int nrhs,
               float  *A, rocblas_int lda, rocblas_int *ipiv,
               float  *B, rocblas_int ldb, rocblas_int *info) override;
    void dgesv(rocblas_int n, rocblas_int nrhs,
               double *A, rocblas_int lda, rocblas_int *ipiv,
               double *B, rocblas_int ldb, rocblas_int *info) override;
    void cgesv(rocblas_int n, rocblas_int nrhs,
               rocblas_float_complex *A, rocblas_int lda, rocblas_int *ipiv,
               rocblas_float_complex *B, rocblas_int ldb, rocblas_int *info) override;
    void zgesv(rocblas_int n, rocblas_int nrhs,
               rocblas_double_complex *A, rocblas_int lda, rocblas_int *ipiv,
               rocblas_double_complex *B, rocblas_int ldb, rocblas_int *info) override;

    void sgetri(rocblas_int n,
                float  *A, rocblas_int lda, const rocblas_int *ipiv,
                rocblas_int *info) override;
    void dgetri(rocblas_int n,
                double *A, rocblas_int lda, const rocblas_int *ipiv,
                rocblas_int *info) override;
    void cgetri(rocblas_int n,
                rocblas_float_complex *A, rocblas_int lda, const rocblas_int *ipiv,
                rocblas_int *info) override;
    void zgetri(rocblas_int n,
                rocblas_double_complex *A, rocblas_int lda, const rocblas_int *ipiv,
                rocblas_int *info) override;

    void spotrf(Layout layout, Uplo uplo, rocblas_int n,
                float  *A, rocblas_int lda, rocblas_int *info) override;
    void dpotrf(Layout layout, Uplo uplo, rocblas_int n,
                double *A, rocblas_int lda, rocblas_int *info) override;
    void cpotrf(Layout layout, Uplo uplo, rocblas_int n,
                rocblas_float_complex *A, rocblas_int lda, rocblas_int *info) override;
    void zpotrf(Layout layout, Uplo uplo, rocblas_int n,
                rocblas_double_complex *A, rocblas_int lda, rocblas_int *info) override;

    void spotri(Layout layout, Uplo uplo, rocblas_int n,
                float *A, rocblas_int lda, rocblas_int *info) override;

    void dpotri(Layout layout, Uplo uplo, rocblas_int n,
                double *A, rocblas_int lda, rocblas_int *info) override;

    void spotrs(Layout layout, Uplo uplo, rocblas_int n, rocblas_int nrhs,
                const float  *A, rocblas_int lda,
                float  *B, rocblas_int ldb, rocblas_int *info) override;
    void dpotrs(Layout layout, Uplo uplo, rocblas_int n, rocblas_int nrhs,
                const double *A, rocblas_int lda,
                double *B, rocblas_int ldb, rocblas_int *info) override;

    void sposv(Layout layout, Uplo uplo, rocblas_int n, rocblas_int nrhs,
               float  *A, rocblas_int lda,
               float  *B, rocblas_int ldb, rocblas_int *info) override;
    void dposv(Layout layout, Uplo uplo, rocblas_int n, rocblas_int nrhs,
               double *A, rocblas_int lda,
               double *B, rocblas_int ldb, rocblas_int *info) override;

    void sgeqrf(Layout layout, rocblas_int m, rocblas_int n,
                float  *A, rocblas_int lda, float  *tau,
                rocblas_int *info) override;
    void dgeqrf(Layout layout, rocblas_int m, rocblas_int n,
                double *A, rocblas_int lda, double *tau,
                rocblas_int *info) override;

    void sorgqr(Layout layout, rocblas_int m, rocblas_int n, rocblas_int k,
                float  *A, rocblas_int lda, const float  *tau,
                rocblas_int *info) override;
    void dorgqr(Layout layout, rocblas_int m, rocblas_int n, rocblas_int k,
                double *A, rocblas_int lda, const double *tau,
                rocblas_int *info) override;

    void sgels(Layout layout, Transpose trans,
               rocblas_int m, rocblas_int n, rocblas_int nrhs,
               float  *A, rocblas_int lda,
               float  *B, rocblas_int ldb, rocblas_int *info) override;
    void dgels(Layout layout, Transpose trans,
               rocblas_int m, rocblas_int n, rocblas_int nrhs,
               double *A, rocblas_int lda,
               double *B, rocblas_int ldb, rocblas_int *info) override;

    void ssyev(Layout layout, Job jobz, Uplo uplo, rocblas_int n,
               float  *A, rocblas_int lda, float  *w,
               rocblas_int *info) override;
    void dsyev(Layout layout, Job jobz, Uplo uplo, rocblas_int n,
               double *A, rocblas_int lda, double *w,
               rocblas_int *info) override;

    void sgeev(Layout layout, Job jobvl, Job jobvr, rocblas_int n,
               float  *A, rocblas_int lda,
               float  *wr, float  *wi,
               float  *VL, rocblas_int ldvl,
               float  *VR, rocblas_int ldvr,
               rocblas_int *info) override;
    void dgeev(Layout layout, Job jobvl, Job jobvr, rocblas_int n,
               double *A, rocblas_int lda,
               double *wr, double *wi,
               double *VL, rocblas_int ldvl,
               double *VR, rocblas_int ldvr,
               rocblas_int *info) override;

    void sgesvd(Layout layout, Job jobu, Job jobvt,
                rocblas_int m, rocblas_int n,
                float  *A, rocblas_int lda, float  *s,
                float  *U, rocblas_int ldu,
                float  *VT, rocblas_int ldvt,
                float  *superb, rocblas_int *info) override;
    void dgesvd(Layout layout, Job jobu, Job jobvt,
                rocblas_int m, rocblas_int n,
                double *A, rocblas_int lda, double *s,
                double *U, rocblas_int ldu,
                double *VT, rocblas_int ldvt,
                double *superb, rocblas_int *info) override;

    void strtrs(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                rocblas_int n, rocblas_int nrhs,
                const float  *A, rocblas_int lda,
                float  *B, rocblas_int ldb, rocblas_int *info) override;
    void dtrtrs(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                rocblas_int n, rocblas_int nrhs,
                const double *A, rocblas_int lda,
                double *B, rocblas_int ldb, rocblas_int *info) override;

private:
    rocsolver_handle m_handle;

    template<typename T>
    void getrf_impl(rocblas_int m, rocblas_int n,
                    T *A, rocblas_int lda,
                    rocblas_int *ipiv, rocblas_int *info);

    template<typename T>
    void getrs_impl(Transpose trans, rocblas_int n, rocblas_int nrhs,
                    const T *A, rocblas_int lda,
                    const rocblas_int *ipiv,
                    T *B, rocblas_int ldb, rocblas_int *info);
    
    template<typename T>
    void gesv_impl(rocblas_int n, rocblas_int nrhs,
                   T *A, rocblas_int lda, rocblas_int *ipiv,
                   T *B, rocblas_int ldb, rocblas_int *info);
    
    template<typename T>
    void getri_impl(rocblas_int n,
                    T *A, rocblas_int lda, const rocblas_int *ipiv,
                    rocblas_int *info);

    template<typename T>
    void potrf_impl(Uplo uplo, rocblas_int n,
                    T *A, rocblas_int lda, rocblas_int *info);

    template<typename T>
    void potri_impl(Uplo uplo, rocblas_int n,
                    T *A, rocblas_int lda, rocblas_int *info);
    
	template<typename T>
	void potrs_impl(Uplo uplo, rocblas_int n, rocblas_int nrhs,
               		const T *A, rocblas_int lda,
                	T *B, rocblas_int ldb,
               		rocblas_int *info);

	template<typename T>
    void gesvd_impl(Job jobu, Job jobvt, rocblas_int m, rocblas_int n,
                    T *A, rocblas_int lda, T *s,
                    T *U, rocblas_int ldu,
                    T *VT, rocblas_int ldvt, rocblas_int *info);

    template<typename T>
    void syev_impl(Job jobz, Uplo uplo, rocblas_int n,
                   T *A, rocblas_int lda, T *w, rocblas_int *info);
};

}
