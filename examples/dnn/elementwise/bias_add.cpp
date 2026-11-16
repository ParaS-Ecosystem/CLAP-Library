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

    clap::TensorDesc xd(clap::DataType::Float32, {1, 3, 2, 2});
    clap::TensorDesc bd(clap::DataType::Float32, {3});

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forBinary(clap::DnnOperation::BiasAdd, xd, bd, xd),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "BiasAdd supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> bias = {100, 200, 300};
    void* d_x = device.uploadVector(x);
    void* d_b = device.uploadVector(bias);
    void* d_y = device.allocate(xd.bytes());
    backend->biasAdd(xd, d_x, bd, d_b, xd, d_y, device.ctx);

    const auto y = device.downloadVector<float>(d_y, x.size());
    std::cout << "x + per-channel bias {100, 200, 300}: ";
    for (float v : y) std::cout << v << ' ';
    std::cout << '\n';
}
