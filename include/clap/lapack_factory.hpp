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

class ILapackBackend {
public:
    virtual ~ILapackBackend() = default;

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

    virtual void sgetri(lapack_int n,
                        float  *A, lapack_int lda,
                        const lapack_int *ipiv,
                        lapack_int *info) = 0;

    virtual void dgetri(lapack_int n,
                        double *A, lapack_int lda,
                        const lapack_int *ipiv,
                        lapack_int *info) = 0;

    virtual void spotrf(Layout layout, Uplo uplo,
                        lapack_int n,
                        float  *A, lapack_int lda,
                        lapack_int *info) = 0;

    virtual void dpotrf(Layout layout, Uplo uplo,
                        lapack_int n,
                        double *A, lapack_int lda,
                        lapack_int *info) = 0;
                        
    virtual void spotri(Layout layout, Uplo uplo,
                        lapack_int n,
                        float *A, lapack_int lda,
                        lapack_int *info) = 0;

    virtual void dpotri(Layout layout, Uplo uplo,
                        lapack_int n,
                        double *A, lapack_int lda,
                        lapack_int *info) = 0;
    
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

class LapackFactory {
public:
    static std::unique_ptr<ILapackBackend>
    create(BackendType default_backend = BackendType::CPU);
};

}
