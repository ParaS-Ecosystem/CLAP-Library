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
