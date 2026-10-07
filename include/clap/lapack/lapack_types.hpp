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
#include <complex>
#include <cstdint>

typedef int lapack_int;

typedef enum {
    CUSOLVER_STATUS_SUCCESS                 = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED         = 1,
    CUSOLVER_STATUS_ALLOC_FAILED            = 2,
    CUSOLVER_STATUS_INVALID_VALUE           = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH           = 4,
    CUSOLVER_STATUS_MAPPING_ERROR           = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED        = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR          = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED           = 9,
    CUSOLVER_STATUS_ZERO_PIVOT              = 10,
    CUSOLVER_STATUS_INVALID_LICENSE         = 11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID      = 13,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR      = 14,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED       = 15,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE        = 16,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 17,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 18,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 19,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR     = 20,
    CUSOLVER_STATUS_INVALID_WORKSPACE       = 21,
} cusolverStatus_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER_SOLVER = 0,
    CUBLAS_FILL_MODE_UPPER_SOLVER = 1,
    CUBLAS_FILL_MODE_FULL_SOLVER  = 2,
} cusolverFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT_SOLVER = 0,
    CUBLAS_DIAG_UNIT_SOLVER     = 1,
} cusolverDiagType_t;

typedef enum {
    CUBLAS_SIDE_LEFT_SOLVER  = 0,
    CUBLAS_SIDE_RIGHT_SOLVER = 1,
} cusolverSideMode_t;

typedef enum {
    CUBLAS_OP_N_SOLVER = 0,
    CUBLAS_OP_T_SOLVER = 1,
    CUBLAS_OP_C_SOLVER = 2,
} cusolverOperation_t;

typedef enum {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR   = 1,
} cusolverEigMode_t;

typedef enum {
    CUSOLVER_EIG_RANGE_ALL = 0,
    CUSOLVER_EIG_RANGE_I   = 1,
    CUSOLVER_EIG_RANGE_V   = 2,
} cusolverEigRange_t;

typedef struct cusolverDnContext *cusolverDnHandle_t;

typedef enum {
    rocblas_status_success_solver         = 0,
    rocblas_status_invalid_handle_solver  = 1,
    rocblas_status_not_implemented_solver = 2,
    rocblas_status_invalid_pointer_solver = 3,
    rocblas_status_invalid_size_solver    = 4,
    rocblas_status_memory_error_solver    = 5,
    rocblas_status_internal_error_solver  = 6,
} rocsolver_status;

typedef int rocblas_int_solver;
typedef struct _rocblas_handle *rocsolver_handle;

typedef enum {
    rocblas_svect_all_solver       = 191,
    rocblas_svect_singular_solver  = 192,
    rocblas_svect_overwrite_solver = 193,
    rocblas_svect_none_solver      = 194,
} rocblas_svect_solver;

typedef enum {
    rocblas_outofplace_solver = 201,
    rocblas_inplace_solver    = 202,
} rocblas_workmode_solver;

typedef enum {
    rocblas_evect_original_solver    = 211,
    rocblas_evect_tridiagonal_solver = 212,
    rocblas_evect_none_solver        = 213,
} rocblas_evect_solver;

namespace clap {

enum class Job {
    NoVec  = 'N',
    Vec    = 'V',
    Update = 'U',
    All    = 'A',
    Some   = 'S',
    Over   = 'O',
};

enum class Range {
    All     = 'A',
    VRange  = 'V',
    IRange  = 'I',
};

enum class Norm {
    One  = '1',
    Inf  = 'I',
    Fro  = 'F',
    Max  = 'M',
};

}
