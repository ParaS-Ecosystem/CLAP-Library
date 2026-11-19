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

    const std::int64_t N = 1, C = 3, H = 2, W = 4;
    clap::TensorDesc xd(clap::DataType::Float32, {N, C, H, W});
    clap::TensorDesc yd(clap::DataType::Float32, {N, H, W, C});

    const auto support = backend->supports(clap::DnnCapabilityQuery::forNchwToNhwc(xd, yd), device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "nchwToNhwc supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x(xd.elements());
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i);
    void* d_x = device.uploadVector(x);
    void* d_y = device.allocate(yd.bytes());
    backend->nchwToNhwc(xd, d_x, yd, d_y, device.ctx);

    const auto y = device.downloadVector<float>(d_y, x.size());
    std::cout << "NHWC: ";
    for (float v : y) std::cout << v << ' ';
    std::cout << '\n';
}
