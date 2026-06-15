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
