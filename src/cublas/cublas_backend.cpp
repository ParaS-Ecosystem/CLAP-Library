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

#include "clap/cublas_backend.hpp"
#include "../include/clap/dyn_backends.hpp"

namespace clap {

CuBlasBackend::CuBlasBackend() {
  if (!dyn::loadCudaAndCublas()) {
    throw std::runtime_error("Failed to load CUDA/cuBLAS dynamically.");
  }
  if (clap_cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle.");
  }
}
CuBlasBackend::~CuBlasBackend() { clap_cublasDestroy(handle); }
} // namespace clap
