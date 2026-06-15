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
