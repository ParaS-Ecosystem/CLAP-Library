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

#include "clap/lapack_factory.hpp"
#include "clap/lapack/lapack/openblas_lapack_backend.hpp"
#include "clap/lapack/cusolver/cusolver_backend.hpp"
#include "clap/lapack/rocsolver/rocsolver_backend.hpp"
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace clap {

std::unique_ptr<ILapackBackend>
LapackFactory::create(BackendType default_backend) {

    BackendType selected = default_backend;

    // Check environment variable (same as BlasFactory)
    const char *env_backend = std::getenv("CLAP_BACKEND");
    if (env_backend) {
        std::string be(env_backend);
        if      (be == "CUDA") selected = BackendType::CUDA;
        else if (be == "AMD") selected = BackendType::AMD;
        else if (be == "CPU")  selected = BackendType::CPU;
    } else {
        // Check command-line args via /proc/self/cmdline
        std::ifstream cmdline("/proc/self/cmdline");
        if (cmdline.is_open()) {
            std::string arg;
            while (std::getline(cmdline, arg, '\0')) {
                if      (arg == "-cuda") { selected = BackendType::CUDA; break; }
                else if (arg == "-rocm") { selected = BackendType::AMD; break; }
                else if (arg == "-cpu")  { selected = BackendType::CPU;  break; }
            }
        }
    }

    if (selected == BackendType::CPU) {
        return std::make_unique<OpenBlasLapackBackend>();
    }
    else if (selected == BackendType::CUDA) {
        return std::make_unique<CuSolverBackend>();
    }
    else if (selected == BackendType::AMD) {
        return std::make_unique<RocSolverBackend>();
    }

    throw std::invalid_argument("Unsupported LAPACK backend requested.");
}

} // namespace clap
