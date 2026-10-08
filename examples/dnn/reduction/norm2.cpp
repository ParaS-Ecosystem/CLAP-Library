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

    clap::TensorDesc x_desc(clap::DataType::Float32, {1, 2, 3, 4});
    clap::TensorDesc y_desc(clap::DataType::Float32, {1, 2, 1, 1});

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forReduction(clap::DnnOperation::TensorReduceNorm2, x_desc, y_desc),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "TensorReduceNorm2 supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x = {
        1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12,
        0.5f, 1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f, 7.5f, -8.5f, 9.5f, -10.5f, 11.5f};

    void* d_x = device.uploadVector(x);
    void* d_y = device.allocate(y_desc.bytes());

    backend->tensorReduceNorm2(x_desc, d_x, y_desc, d_y, device.ctx);

    const auto y = device.downloadVector<float>(d_y, 2);
    std::cout << "result  : " << y[0] << ' ' << y[1] << '\n';

    std::vector<float> legacy(2);
    backend->tensorReduceNorm2(x_desc, x.data(), y_desc, legacy.data());
    std::cout << "legacy  : " << legacy[0] << ' ' << legacy[1] << '\n';
}
