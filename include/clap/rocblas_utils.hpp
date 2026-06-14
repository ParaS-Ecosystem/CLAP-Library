// -----------------------------------------------------------------------------
// This file is part of the CLAP library — an implementation of
// high-performance linear algebra routines based on the BLAS standard.
//
// CLAP provides a unified math interface across architectures,
// enabling portable and optimized performance on CPUs and GPUs.
//
// Copyright (C) The CLAP Contributors
//
// CLAP is released under the GNU Lesser General Public License (LGPL) v3.
// See the file LICENSE in the project root for full license details.
// -----------------------------------------------------------------------------

#include "blas_factory.hpp"
#include "dyn_backends.hpp"

namespace clap {

typedef int hipError_t;

typedef std::complex<float> roc_complex;
typedef std::complex<double> roc_DoubleComplex;

static rocblas_operation to_rocblas_trans(Transpose t) {
  switch (t) {
  case Transpose::NoTrans:
    return rocblas_operation_none;
  case Transpose::Trans:
    return rocblas_operation_transpose;
  case Transpose::ConjTrans:
    return rocblas_operation_conjugate_transpose;
  default:
    throw std::invalid_argument("Invalid Transpose");
  }
}

static rocblas_fill to_rocblas_uplo(Uplo u) {
  return (u == Uplo::Upper) ? rocblas_fill_upper : rocblas_fill_lower;
}

static rocblas_side to_rocblas_side(Side s) {
  return (s == Side::Left) ? rocblas_side_left : rocblas_side_right;
}

static rocblas_diagonal to_rocblas_diag(Diag d) {
  return (d == Diag::Unit) ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;
}

} // namespace clap
