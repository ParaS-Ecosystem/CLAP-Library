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


#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap_example::Device device(*backend);

    clap::TensorDesc d(clap::DataType::Float32, {2, 3, 4});

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forOperation(clap::DnnOperation::TensorFill, d),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "TensorFill supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    void* d_y = device.allocate(d.bytes());
    backend->tensorFill(d, d_y, 0.5f, device.ctx);

    const auto y = device.downloadVector<float>(d_y, d.elements());
    std::cout << "filled: ";
    for (float v : y) std::cout << v << ' ';
    std::cout << '\n';
}
