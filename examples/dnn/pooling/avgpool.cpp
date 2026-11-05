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

    clap::TensorDesc xd(clap::DataType::Float32, {1, 1, 4, 4});
    clap::TensorDesc yd(clap::DataType::Float32, {1, 1, 4, 4});
    clap::PoolingDesc pool;
    pool.window_h = 3;
    pool.window_w = 3;
    pool.pad_h = 1;
    pool.pad_w = 1;
    pool.stride_h = 1;
    pool.stride_w = 1;

    std::cout << "Backend: " << backend->name() << "\n";

    std::vector<float> x(16);
    for (int i = 0; i < 16; ++i)
        x[i] = static_cast<float>(i + 1);

    void* d_x = device.uploadVector(x);
    void* d_y = device.allocate(yd.bytes());

    for (auto mode : {clap::PoolingMode::Average, clap::PoolingMode::AverageExcludePadding}) {
        pool.mode = mode;
        backend->poolingForward(pool, xd, d_x, yd, d_y, device.ctx);
        const auto y = device.downloadVector<float>(d_y, 16);
        std::cout << (mode == clap::PoolingMode::Average ? "include padding: " : "exclude padding: ");
        for (float v : y) std::cout << v << ' ';
        std::cout << '\n';
    }
}
