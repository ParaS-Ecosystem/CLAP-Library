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

    clap::TensorDesc d(clap::DataType::Float32, {2, 2, 2});

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forBinary(clap::DnnOperation::TensorSubtract, d, d, d),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "TensorSubtract supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> a = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> b = {1, 2, 3, 4, 5, 6, 7, 8};
    void* d_a = device.uploadVector(a);
    void* d_b = device.uploadVector(b);
    void* d_y = device.allocate(d.bytes());
    backend->tensorSubtract(d, d_a, d, d_b, d, d_y, device.ctx);

    const auto y = device.downloadVector<float>(d_y, a.size());
    std::cout << "a - b: ";
    for (float v : y) std::cout << v << ' ';
    std::cout << '\n';
}
