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

#pragma once

#include "clap/dnn_factory.hpp"
#include "clap/dnn_types.hpp"
#include "clap/idnn_backend.hpp"

namespace clap {

inline std::unique_ptr<IDnnBackend> createDnnBackend()
{
#if defined(gpu)
    return DnnFactory::create(DnnBackendType::GPU);
#elif defined(cpu)
    return DnnFactory::create(DnnBackendType::CPU);
#else
    return DnnFactory::create(DnnBackendType::CPU);
#endif
}

}
