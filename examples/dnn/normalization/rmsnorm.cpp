#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <iostream>
#include <vector>

// RMSNorm (dedicated operation, not LayerNorm) over the hidden dimension of a
// [B, S, H] tensor, forward in training mode (saved_rstd) and backward.
int main() {
    auto backend = clap::createDnnBackend();
    clap_example::Device device(*backend);

    const std::int64_t B = 2, S = 2, H = 4;
    clap::TensorDesc d(clap::DataType::Float32, {B, S, H});
    clap::TensorDesc wd(clap::DataType::Float32, {H});
    clap::RmsNormDesc norm;
    norm.epsilon = 1.0e-6;

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forRmsNorm(clap::DnnOperation::RmsNormForward, norm, d, wd),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "RMSNorm supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x = {1, 2, 3, 4, 2, 4, 6, 8, -1, 0, 1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> w = {1.0f, 0.5f, 2.0f, 1.0f};
    std::vector<float> dy(x.size(), 1.0f);

    void* d_x = device.uploadVector(x);
    void* d_w = device.uploadVector(w);
    void* d_dy = device.uploadVector(dy);
    void* d_y = device.allocate(d.bytes());
    void* d_rstd = device.allocate(B * S * sizeof(float));
    void* d_dx = device.allocate(d.bytes());
    void* d_dw = device.allocate(wd.bytes());

    backend->rmsNormForward(norm, d, d_x, wd, d_w, d, d_y, d_rstd, device.ctx);

    const auto y = device.downloadVector<float>(d_y, x.size());
    const auto rstd = device.downloadVector<float>(d_rstd, B * S);

    std::cout << "forward : ";
    for (float v : y) std::cout << v << ' ';
    std::cout << "\nrstd    : ";
    for (float v : rstd) std::cout << v << ' ';
    std::cout << '\n';

    const auto backward = backend->supports(
        clap::DnnCapabilityQuery::forRmsNorm(clap::DnnOperation::RmsNormBackward, norm, d, wd),
        device.ctx);
    if (!backward) {
        std::cout << "backward: unsupported natively (" << backward.reason << ")\n";
        return 0;
    }

    backend->rmsNormBackward(norm, d, d_x, d, d_dy, wd, d_w, d_rstd, d, d_dx, d_dw, device.ctx);

    const auto dx = device.downloadVector<float>(d_dx, x.size());
    const auto dw = device.downloadVector<float>(d_dw, H);

    std::cout << "dx      : ";
    for (float v : dx) std::cout << v << ' ';
    std::cout << "\ndweight : ";
    for (float v : dw) std::cout << v << ' ';
    std::cout << '\n';
}
