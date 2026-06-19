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

// Macros for easy calling

// CUBLAS

#define clap_cudaMalloc clap::dyn::p_cudaMalloc
#define clap_cudaFree clap::dyn::p_cudaFree
#define clap_cudaMemcpy clap::dyn::p_cudaMemcpy
#define clap_cublasSetPointerMode clap::dyn::p_cublasSetPointerMode_v2
#define clap_cudaDeviceSynchronize clap::dyn::p_cudaDeviceSynchronize
#define clap_cublasCreate clap::dyn::p_cublasCreate_v2
#define clap_cublasDestroy clap::dyn::p_cublasDestroy_v2
#define clap_cudaMemset clap::dyn::p_cudaMemset

// level 1

#define clap_cublasScopy clap::dyn::p_cublasScopy_v2
#define clap_cublasDcopy clap::dyn::p_cublasDcopy_v2

#define clap_cublasSscal clap::dyn::p_cublasSscal_v2
#define clap_cublasDscal clap::dyn::p_cublasDscal_v2

#define clap_cublasSaxpy clap::dyn::p_cublasSaxpy_v2
#define clap_cublasDaxpy clap::dyn::p_cublasDaxpy_v2

#define clap_cublasSswap clap::dyn::p_cublasSswap_v2
#define clap_cublasDswap clap::dyn::p_cublasDswap_v2

#define clap_cublasSdot clap::dyn::p_cublasSdot_v2
#define clap_cublasDdot clap::dyn::p_cublasDdot_v2

// level 2

#define clap_cublasSgemv clap::dyn::p_cublasSgemv_v2
#define clap_cublasDgemv clap::dyn::p_cublasDgemv_v2

#define clap_cublasSsymv clap::dyn::p_cublasSsymv_v2
#define clap_cublasDsymv clap::dyn::p_cublasDsymv_v2

#define clap_cublasStrmv clap::dyn::p_cublasStrmv_v2
#define clap_cublasDtrmv clap::dyn::p_cublasDtrmv_v2

#define clap_cublasStrsv clap::dyn::p_cublasStrsv_v2
#define clap_cublasDtrsv clap::dyn::p_cublasDtrsv_v2

#define clap_cublasSsbmv clap::dyn::p_cublasSsbmv_v2
#define clap_cublasDsbmv clap::dyn::p_cublasDsbmv_v2

#define clap_cublasSger clap::dyn::p_cublasSger_v2

// Level 3

#define clap_cublasSgemm clap::dyn::p_cublasSgemm_v2
#define clap_cublasDgemm clap::dyn::p_cublasDgemm_v2

#define clap_cublasSsymm clap::dyn::p_cublasSsymm_v2
#define clap_cublasDsymm clap::dyn::p_cublasDsymm_v2

#define clap_cublasSsyrk clap::dyn::p_cublasSsyrk_v2
#define clap_cublasDsyrk clap::dyn::p_cublasDsyrk_v2

#define clap_cublasStrmm clap::dyn::p_cublasStrmm_v2
#define clap_cublasDtrmm clap::dyn::p_cublasDtrmm_v2

#define clap_cublasStrsm clap::dyn::p_cublasStrsm_v2
#define clap_cublasDtrsm clap::dyn::p_cublasDtrsm_v2

// OPENBLAS

// Level 1

#define clap_cblas_scopy clap::dyn::p_cblas_scopy
#define clap_cblas_dcopy clap::dyn::p_cblas_dcopy

#define clap_cblas_sscal clap::dyn::p_cblas_sscal
#define clap_cblas_dscal clap::dyn::p_cblas_dscal

#define clap_cblas_saxpy clap::dyn::p_cblas_saxpy
#define clap_cblas_daxpy clap::dyn::p_cblas_daxpy

#define clap_cblas_sswap clap::dyn::p_cblas_sswap
#define clap_cblas_dswap clap::dyn::p_cblas_dswap

#define clap_cblas_sdot clap::dyn::p_cblas_sdot
#define clap_cblas_ddot clap::dyn::p_cblas_ddot

// level 2

#define clap_cblas_sgemv clap::dyn::p_cblas_sgemv
#define clap_cblas_dgemv clap::dyn::p_cblas_dgemv

#define clap_cblas_ssymv clap::dyn::p_cblas_ssymv
#define clap_cblas_dsymv clap::dyn::p_cblas_dsymv

#define clap_cblas_strmv clap::dyn::p_cblas_strmv
#define clap_cblas_dtrmv clap::dyn::p_cblas_dtrmv

#define clap_cblas_strsv clap::dyn::p_cblas_strsv
#define clap_cblas_dtrsv clap::dyn::p_cblas_dtrsv

#define clap_cblas_ssbmv clap::dyn::p_cblas_ssbmv
#define clap_cblas_dsbmv clap::dyn::p_cblas_dsbmv

#define clap_cblas_sger clap::dyn::p_cblas_sger

// Level 3

#define clap_cblas_sgemm clap::dyn::p_cblas_sgemm
#define clap_cblas_dgemm clap::dyn::p_cblas_dgemm

#define clap_cblas_ssymm clap::dyn::p_cblas_ssymm
#define clap_cblas_dsymm clap::dyn::p_cblas_dsymm

#define clap_cblas_ssyrk clap::dyn::p_cblas_ssyrk
#define clap_cblas_dsyrk clap::dyn::p_cblas_dsyrk

#define clap_cblas_strmm clap::dyn::p_cblas_strmm
#define clap_cblas_dtrmm clap::dyn::p_cblas_dtrmm

#define clap_cblas_strsm clap::dyn::p_cblas_strsm
#define clap_cblas_dtrsm clap::dyn::p_cblas_dtrsm

// ROCBLAS

#define clap_hipMalloc clap::dyn::p_hipMalloc
#define clap_hipFree clap::dyn::p_hipFree
#define clap_hipMemcpy clap::dyn::p_hipMemcpy

#define clap_rocblas_create_handle clap::dyn::p_rocblas_create_handle
#define clap_rocblas_destroy_handle clap::dyn::p_rocblas_destroy_handle

// Level 1

#define clap_rocblas_saxpy clap::dyn::p_rocblas_saxpy
#define clap_rocblas_daxpy clap::dyn::p_rocblas_daxpy

#define clap_rocblas_scopy clap::dyn::p_rocblas_scopy
#define clap_rocblas_dcopy clap::dyn::p_rocblas_dcopy

#define clap_rocblas_sdot clap::dyn::p_rocblas_sdot
#define clap_rocblas_ddot clap::dyn::p_rocblas_ddot

#define clap_rocblas_sscal clap::dyn::p_rocblas_sscal
#define clap_rocblas_dscal clap::dyn::p_rocblas_dscal

#define clap_rocblas_sswap clap::dyn::p_rocblas_sswap
#define clap_rocblas_dswap clap::dyn::p_rocblas_dswap

// Level 2

#define clap_rocblasSgemv clap::dyn::p_rocblas_sgemv
#define clap_rocblasDgemv clap::dyn::p_rocblas_dgemv

#define clap_rocblasSsymv clap::dyn::p_rocblas_ssymv
#define clap_rocblasDsymv clap::dyn::p_rocblas_dsymv

#define clap_rocblasStrmv clap::dyn::p_rocblas_strmv
#define clap_rocblasDtrmv clap::dyn::p_rocblas_dtrmv

#define clap_rocblasStrsv clap::dyn::p_rocblas_strsv
#define clap_rocblasDtrsv clap::dyn::p_rocblas_dtrsv

#define clap_rocblasSsbmv clap::dyn::p_rocblas_ssbmv
#define clap_rocblasDsbmv clap::dyn::p_rocblas_dsbmv

#define clap_rocblasSger clap::dyn::p_rocblas_sger

// Level 3

#define clap_rocblas_sgemm clap::dyn::p_rocblas_sgemm
#define clap_rocblas_dgemm clap::dyn::p_rocblas_dgemm

#define clap_rocblas_ssymm clap::dyn::p_rocblas_ssymm
#define clap_rocblas_dsymm clap::dyn::p_rocblas_dsymm

#define clap_rocblas_ssyrk clap::dyn::p_rocblas_ssyrk
#define clap_rocblas_dsyrk clap::dyn::p_rocblas_dsyrk

#define clap_rocblas_strmm clap::dyn::p_rocblas_strmm
#define clap_rocblas_dtrmm clap::dyn::p_rocblas_dtrmm

#define clap_rocblas_strsm clap::dyn::p_rocblas_strsm
#define clap_rocblas_dtrsm clap::dyn::p_rocblas_dtrsm
