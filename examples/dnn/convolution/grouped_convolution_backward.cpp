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

    const std::int64_t N = 1, C = 4, H = 4, W = 4, K = 6, G = 2;
    clap::TensorDesc xd(clap::DataType::Float32, {N, C, H, W});
    clap::TensorDesc wd(clap::DataType::Float32, {K, C / G, 3, 3});
    clap::TensorDesc yd(clap::DataType::Float32, {N, K, H, W});
    clap::ConvolutionDesc conv;
    conv.pad_h = conv.pad_w = 1;
    conv.groups = G;

    const auto data = backend->supports(
        clap::DnnCapabilityQuery::forConvolution(clap::DnnOperation::ConvolutionBackwardData, conv, xd, wd, yd), device.ctx);
    const auto weights = backend->supports(
        clap::DnnCapabilityQuery::forConvolution(clap::DnnOperation::ConvolutionBackwardWeights, conv, xd, wd, yd), device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "grouped convolutionBackwardData supported: " << (data ? "yes" : "no") << " " << data.reason << "\n";
    std::cout << "grouped convolutionBackwardWeights supported: " << (weights ? "yes" : "no") << " " << weights.reason << "\n";
    if (!data || !weights)
        return 0;

    std::vector<float> x(xd.elements()), w(wd.elements()), dy(yd.elements());
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i % 5) - 2.0f;
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>(i % 3) * 0.5f;
    for (std::size_t i = 0; i < dy.size(); ++i)
        dy[i] = (i % 2) ? 1.0f : -1.0f;
    void* d_x = device.uploadVector(x);
    void* d_w = device.uploadVector(w);
    void* d_dy = device.uploadVector(dy);
    void* d_dx = device.allocate(xd.bytes());
    void* d_dw = device.allocate(wd.bytes());

    backend->convolutionBackwardData(yd, d_dy, wd, d_w, conv, xd, d_dx, device.ctx);
    backend->convolutionBackwardWeights(xd, d_x, yd, d_dy, conv, wd, d_dw, device.ctx);

    const auto dx = device.downloadVector<float>(d_dx, xd.elements());
    const auto dw = device.downloadVector<float>(d_dw, wd.elements());
    std::cout << "dx: ";
    for (float v : dx) std::cout << v << ' ';
    std::cout << "\ndw: ";
    for (float v : dw) std::cout << v << ' ';
    std::cout << '\n';
}
