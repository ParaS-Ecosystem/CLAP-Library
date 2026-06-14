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

#include "clap/openblas_backend.hpp"
#include "../include/clap/dyn_backends.hpp"

namespace clap {

OpenBlasBackend::OpenBlasBackend() {
  std::cout << "Using OpenBLAS backend\n";
  if (!dyn::loadOpenBlas())
    throw std::runtime_error("OpenBLAS load failed");
}

} // namespace clap
