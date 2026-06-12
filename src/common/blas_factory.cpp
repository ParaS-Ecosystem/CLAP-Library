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

#include "clap/blas_factory.hpp"
#include "clap/cublas_backend.hpp"
#include "clap/openblas_backend.hpp"
#include "clap/rocblas_backend.hpp"
#include <fstream>

namespace clap {

std::unique_ptr<IBlasBackend> BlasFactory::create(BackendType default_backend) {
  BackendType selected = default_backend;

  const char *env_backend = std::getenv("CLAP_BACKEND");

  if (env_backend) {
    std::string be(env_backend);
    if (be == "CUDA")
      selected = BackendType::CUDA;
    else if (be == "ROCM")
      selected = BackendType::ROCM;
    else if (be == "CPU")
      selected = BackendType::CPU;

  } else {

    std::ifstream cmdline("/proc/self/cmdline");
    if (cmdline.is_open()) {
      std::string arg;

      while (std::getline(cmdline, arg, '\0')) {
        if (arg == "-cuda") {
          selected = BackendType::CUDA;
          break;
        }
        if (arg == "-rocm") {
          selected = BackendType::ROCM;
          break;
        }
        if (arg == "-cpu") {
          selected = BackendType::CPU;
          break;
        }
      }
    }
  }

  if (selected == BackendType::CPU) {
    return std::make_unique<OpenBlasBackend>();
  } else if (selected == BackendType::CUDA) {
    return std::make_unique<CuBlasBackend>();
  } else {
    return std::make_unique<RocBlasBackend>();
  }

  throw std::invalid_argument("Unsupported backend requested.");
}

} // namespace clap
