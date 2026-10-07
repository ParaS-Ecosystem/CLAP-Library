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

#define clap_LAPACKE_sgetrf  clap::dyn_lapack::p_LAPACKE_sgetrf
#define clap_LAPACKE_dgetrf  clap::dyn_lapack::p_LAPACKE_dgetrf
#define clap_LAPACKE_sgetrs  clap::dyn_lapack::p_LAPACKE_sgetrs
#define clap_LAPACKE_dgetrs  clap::dyn_lapack::p_LAPACKE_dgetrs
#define clap_LAPACKE_sgesv   clap::dyn_lapack::p_LAPACKE_sgesv
#define clap_LAPACKE_dgesv   clap::dyn_lapack::p_LAPACKE_dgesv
#define clap_LAPACKE_sgetri  clap::dyn_lapack::p_LAPACKE_sgetri
#define clap_LAPACKE_dgetri  clap::dyn_lapack::p_LAPACKE_dgetri

#define clap_LAPACKE_spotrf  clap::dyn_lapack::p_LAPACKE_spotrf
#define clap_LAPACKE_dpotrf  clap::dyn_lapack::p_LAPACKE_dpotrf
#define clap_LAPACKE_spotri  clap::dyn_lapack::p_LAPACKE_spotri
#define clap_LAPACKE_dpotri  clap::dyn_lapack::p_LAPACKE_dpotri
#define clap_LAPACKE_spotrs  clap::dyn_lapack::p_LAPACKE_spotrs
#define clap_LAPACKE_dpotrs  clap::dyn_lapack::p_LAPACKE_dpotrs
#define clap_LAPACKE_sposv   clap::dyn_lapack::p_LAPACKE_sposv
#define clap_LAPACKE_dposv   clap::dyn_lapack::p_LAPACKE_dposv

#define clap_LAPACKE_sgeqrf  clap::dyn_lapack::p_LAPACKE_sgeqrf
#define clap_LAPACKE_dgeqrf  clap::dyn_lapack::p_LAPACKE_dgeqrf
#define clap_LAPACKE_sorgqr  clap::dyn_lapack::p_LAPACKE_sorgqr
#define clap_LAPACKE_dorgqr  clap::dyn_lapack::p_LAPACKE_dorgqr
#define clap_LAPACKE_sgels   clap::dyn_lapack::p_LAPACKE_sgels
#define clap_LAPACKE_dgels   clap::dyn_lapack::p_LAPACKE_dgels

#define clap_LAPACKE_ssyev   clap::dyn_lapack::p_LAPACKE_ssyev
#define clap_LAPACKE_dsyev   clap::dyn_lapack::p_LAPACKE_dsyev
#define clap_LAPACKE_sgeev   clap::dyn_lapack::p_LAPACKE_sgeev
#define clap_LAPACKE_dgeev   clap::dyn_lapack::p_LAPACKE_dgeev

#define clap_LAPACKE_sgesvd  clap::dyn_lapack::p_LAPACKE_sgesvd
#define clap_LAPACKE_dgesvd  clap::dyn_lapack::p_LAPACKE_dgesvd

#define clap_LAPACKE_strtrs  clap::dyn_lapack::p_LAPACKE_strtrs
#define clap_LAPACKE_dtrtrs  clap::dyn_lapack::p_LAPACKE_dtrtrs

#define LAPACK_ROW_MAJOR  101
#define LAPACK_COL_MAJOR  102

#define clap_cusolverDnCreate               clap::dyn_lapack::p_cusolverDnCreate
#define clap_cusolverDnDestroy              clap::dyn_lapack::p_cusolverDnDestroy

#define clap_cusolverDnSgetrf_bufferSize    clap::dyn_lapack::p_cusolverDnSgetrf_bufferSize
#define clap_cusolverDnDgetrf_bufferSize    clap::dyn_lapack::p_cusolverDnDgetrf_bufferSize
#define clap_cusolverDnSgetrf               clap::dyn_lapack::p_cusolverDnSgetrf
#define clap_cusolverDnDgetrf               clap::dyn_lapack::p_cusolverDnDgetrf
#define clap_cusolverDnSgetrs               clap::dyn_lapack::p_cusolverDnSgetrs
#define clap_cusolverDnDgetrs               clap::dyn_lapack::p_cusolverDnDgetrs

#define clap_cusolverDnSpotrf_bufferSize    clap::dyn_lapack::p_cusolverDnSpotrf_bufferSize
#define clap_cusolverDnDpotrf_bufferSize    clap::dyn_lapack::p_cusolverDnDpotrf_bufferSize
#define clap_cusolverDnSpotrf               clap::dyn_lapack::p_cusolverDnSpotrf
#define clap_cusolverDnDpotrf               clap::dyn_lapack::p_cusolverDnDpotrf

#define clap_cusolverDnSpotri_bufferSize    clap::dyn_lapack::p_cusolverDnSpotri_bufferSize
#define clap_cusolverDnDpotri_bufferSize    clap::dyn_lapack::p_cusolverDnDpotri_bufferSize
#define clap_cusolverDnSpotri               clap::dyn_lapack::p_cusolverDnSpotri
#define clap_cusolverDnDpotri               clap::dyn_lapack::p_cusolverDnDpotri

#define clap_cusolverDnSpotrs               clap::dyn_lapack::p_cusolverDnSpotrs
#define clap_cusolverDnDpotrs               clap::dyn_lapack::p_cusolverDnDpotrs

#define clap_cusolverDnSgesvd_bufferSize    clap::dyn_lapack::p_cusolverDnSgesvd_bufferSize
#define clap_cusolverDnDgesvd_bufferSize    clap::dyn_lapack::p_cusolverDnDgesvd_bufferSize
#define clap_cusolverDnSgesvd               clap::dyn_lapack::p_cusolverDnSgesvd
#define clap_cusolverDnDgesvd               clap::dyn_lapack::p_cusolverDnDgesvd

#define clap_cusolverDnSsyevd_bufferSize    clap::dyn_lapack::p_cusolverDnSsyevd_bufferSize
#define clap_cusolverDnDsyevd_bufferSize    clap::dyn_lapack::p_cusolverDnDsyevd_bufferSize
#define clap_cusolverDnSsyevd               clap::dyn_lapack::p_cusolverDnSsyevd
#define clap_cusolverDnDsyevd               clap::dyn_lapack::p_cusolverDnDsyevd

#define clap_rocsolver_sgetrf   clap::dyn_lapack::p_rocsolver_sgetrf
#define clap_rocsolver_dgetrf   clap::dyn_lapack::p_rocsolver_dgetrf
#define clap_rocsolver_sgetrs   clap::dyn_lapack::p_rocsolver_sgetrs
#define clap_rocsolver_dgetrs   clap::dyn_lapack::p_rocsolver_dgetrs
#define clap_rocsolver_spotrf   clap::dyn_lapack::p_rocsolver_spotrf
#define clap_rocsolver_spotri   clap::dyn_lapack::p_rocsolver_spotri
#define clap_rocsolver_dpotri   clap::dyn_lapack::p_rocsolver_dpotri
#define clap_rocsolver_dpotrf   clap::dyn_lapack::p_rocsolver_dpotrf
#define clap_rocsolver_spotrs   clap::dyn_lapack::p_rocsolver_spotrs
#define clap_rocsolver_dpotrs   clap::dyn_lapack::p_rocsolver_dpotrs
#define clap_rocsolver_sgesvd   clap::dyn_lapack::p_rocsolver_sgesvd
#define clap_rocsolver_dgesvd   clap::dyn_lapack::p_rocsolver_dgesvd
#define clap_rocsolver_ssyev    clap::dyn_lapack::p_rocsolver_ssyev
#define clap_rocsolver_dsyev    clap::dyn_lapack::p_rocsolver_dsyev
