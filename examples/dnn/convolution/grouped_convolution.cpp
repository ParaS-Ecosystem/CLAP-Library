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

    const std::int64_t N = 1, C = 4, H = 4, W = 4, K = 4, G = 2;
    clap::TensorDesc xd(clap::DataType::Float32, {N, C, H, W});
    clap::TensorDesc wd(clap::DataType::Float32, {K, C / G, 3, 3});
    clap::TensorDesc yd(clap::DataType::Float32, {N, K, H, W});
    clap::ConvolutionDesc conv;
    conv.pad_h = conv.pad_w = 1;
    conv.groups = G;

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forConvolution(clap::DnnOperation::ConvolutionForward, conv, xd, wd, yd), device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "grouped convolutionForward supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x(xd.elements()), w(wd.elements());
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i % 7);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = (i % 2) ? 0.5f : -0.25f;
    void* d_x = device.uploadVector(x);
    void* d_w = device.uploadVector(w);
    void* d_y = device.allocate(yd.bytes());
    backend->convolutionForward(xd, d_x, wd, d_w, conv, yd, d_y, device.ctx);

    const auto y = device.downloadVector<float>(d_y, yd.elements());
    std::cout << "y (groups = " << G << "): ";
    for (float v : y) std::cout << v << ' ';
    std::cout << '\n';
}
