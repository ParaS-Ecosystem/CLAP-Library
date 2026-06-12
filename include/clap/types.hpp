// -----------------------------------------------------------------------------
// This file is part of the CLAP library — an implementation of
// high-performance linear algebra routines based on the BLAS standard.
//
// CLAP provides a unified math interface across architectures,
// enabling portable and optimized performance on CPUs.
//
// Copyright (C) The CLAP Contributors
//
// CLAP is released under the GNU Lesser General Public License (LGPL) v3.
// See the file LICENSE in the project root for full license details.
// -----------------------------------------------------------------------------

// ==== CUDA & cuBLAS minimal typedefs ====

#include <complex>

typedef int cudaError_t;
typedef std::complex<float> cuComplex;
typedef std::complex<double> cuDoubleComplex;
inline cuComplex make_cuComplex(float real, float imag) {
  return cuComplex(real, imag);
}

inline cuDoubleComplex make_cuDoubleComplex(double real, double imag) {
  return cuDoubleComplex(real, imag);
}
inline float cuCrealf(const cuComplex &c) { return c.real(); }

inline float cuCimagf(const cuComplex &c) { return c.imag(); }

inline double cuCreal(const cuDoubleComplex &c) { return c.real(); }

inline double cuCimag(const cuDoubleComplex &c) { return c.imag(); }

typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14
} cublasStatus_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
  CUBLAS_FILL_MODE_LOWER = 0,
  CUBLAS_FILL_MODE_UPPER = 1
} cublasFillMode_t;

typedef enum {
  CUBLAS_DIAG_NON_UNIT = 0,
  CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;

typedef enum cublasPointerMode_t {
  CUBLAS_POINTER_MODE_HOST = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef struct cublasContext *cublasHandle_t;

// ==== OpenBLAS minimal typedefs ====

typedef enum CBLAS_ORDER {
  CblasRowMajor = 101,
  CblasColMajor = 102
} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;

typedef enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;

//=======rocblas minimal typedefs=======

typedef int hipError_t;
typedef int rocblas_int;

typedef std::complex<float> rocblas_float_complex;
typedef std::complex<double> rocblas_double_complex;

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;

typedef enum rocblas_status {
  rocblas_status_success = 0,
  rocblas_status_invalid_handle = 1,
  rocblas_status_not_implemented = 2,
  rocblas_status_invalid_pointer = 3,
  rocblas_status_invalid_size = 4,
  rocblas_status_memory_error = 5,
  rocblas_status_internal_error = 6,
  rocblas_status_perf_degraded = 7,
  rocblas_status_size_query_mismatch = 8,
  rocblas_status_size_increased = 9,
  rocblas_status_size_unchanged = 10,
  rocblas_status_invalid_value = 11,
  rocblas_status_continue = 12,
  rocblas_status_check_numerics_fail = 13
} rocblas_status;

typedef enum rocblas_operation {
  rocblas_operation_none = 111,
  rocblas_operation_transpose = 112,
  rocblas_operation_conjugate_transpose = 113
} rocblas_operation;

typedef enum rocblas_fill {
  rocblas_fill_upper = 121,
  rocblas_fill_lower = 122,
  rocblas_fill_full = 123
} rocblas_fill;

typedef enum rocblas_diagonal {
  rocblas_diagonal_non_unit = 131,
  rocblas_diagonal_unit = 132
} rocblas_diagonal;

typedef enum rocblas_side {
  rocblas_side_left = 141,
  rocblas_side_right = 142,
  rocblas_side_both = 143
} rocblas_side;

typedef struct _rocblas_handle *rocblas_handle;
