#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap_example::Device device(*backend);

    clap::TensorDesc d(clap::DataType::Float32, {2, 2, 4});
    clap::ActivationDesc silu;
    silu.mode = clap::ActivationMode::SiLU;

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forActivation(clap::DnnOperation::ActivationForward, silu, d),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "SiLU supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x = {-3, -1, 0, 1, 2, -2, 5, 6, 0.5f, -0.5f, 4, -4, 1.5f, -1.5f, 3, -3};
    std::vector<float> dy(x.size(), 1.0f);

    void* d_x = device.uploadVector(x);
    void* d_dy = device.uploadVector(dy);
    void* d_y = device.allocate(d.bytes());
    void* d_dx = device.allocate(d.bytes());

    backend->activationForward(silu, d, d_x, d, d_y, device.ctx);
    backend->activationBackward(silu, d, d_x, d, d_dy, d, d_dx, device.ctx);

    const auto y = device.downloadVector<float>(d_y, x.size());
    const auto dx = device.downloadVector<float>(d_dx, x.size());

    std::cout << "forward : ";
    for (float v : y) std::cout << v << ' ';
    std::cout << "\nbackward: ";
    for (float v : dx) std::cout << v << ' ';
    std::cout << '\n';
}
