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

static CBLAS_ORDER to_cblas_order(Layout l) {
  return l == Layout::RowMajor ? CblasRowMajor : CblasColMajor;
}

static CBLAS_TRANSPOSE to_cblas_trans(Transpose t) {
  if (t == Transpose::NoTrans)
    return CblasNoTrans;
  if (t == Transpose::Trans)
    return CblasTrans;
  return CblasConjTrans;
}

static CBLAS_UPLO to_cblas_uplo(Uplo u) {
  return u == Uplo::Upper ? CblasUpper : CblasLower;
}

static CBLAS_DIAG to_cblas_diag(Diag d) {
  return d == Diag::Unit ? CblasUnit : CblasNonUnit;
}
inline CBLAS_SIDE to_cblas_side(Side s) {
  return (s == Side::Left) ? CblasLeft : CblasRight;
}
} // namespace clap
