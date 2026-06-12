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

#include "blas_factory.hpp"
#include "dyn_backends.hpp"

namespace clap {

inline cublasOperation_t to_cublas_trans(Transpose t) {
  if (t == Transpose::NoTrans)
    return CUBLAS_OP_N;
  if (t == Transpose::Trans)
    return CUBLAS_OP_T;
  return CUBLAS_OP_C;
}
inline cublasFillMode_t to_cublas_uplo(Uplo u) {
  return u == Uplo::Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}
inline cublasDiagType_t to_cublas_diag(Diag d) {
  return d == Diag::Unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
}
inline cublasSideMode_t to_cublas_side(Side s) {
  return (s == Side::Left) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
}

} // namespace clap
