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

    clap::TensorDesc d(clap::DataType::Float32, {2, 2, 4});
    clap::ActivationDesc act;
    act.mode = clap::ActivationMode::ClippedReLU;
    act.alpha = 6.0;

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forActivation(clap::DnnOperation::ActivationForward, act, d),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "ClippedReLU supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x = {-3, -1, -0.25f, 0.5f, 2, -2, 5, 7, 0.75f, -0.5f, 4, -4, 1.5f, -1.5f, 3, 8};
    std::vector<float> dy(x.size(), 1.0f);

    void* d_x = device.uploadVector(x);
    void* d_dy = device.uploadVector(dy);
    void* d_y = device.allocate(d.bytes());
    void* d_dx = device.allocate(d.bytes());

    backend->activationForward(act, d, d_x, d, d_y, device.ctx);
    backend->activationBackward(act, d, d_x, d, d_dy, d, d_dx, device.ctx);

    const auto y = device.downloadVector<float>(d_y, x.size());
    const auto dx = device.downloadVector<float>(d_dx, x.size());

    std::cout << "forward : ";
    for (float v : y) std::cout << v << ' ';
    std::cout << "\nbackward: ";
    for (float v : dx) std::cout << v << ' ';
    std::cout << '\n';
}
