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
