#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <iostream>
#include <vector>

// LogSoftmax forward/backward along the last axis of a [Tokens, Experts]
// router-logit tensor.  Regular softmax is unchanged; LogSoftmax is selected
// with SoftmaxDesc::log_softmax.
int main() {
    auto backend = clap::createDnnBackend();
    clap_example::Device device(*backend);

    clap::TensorDesc d(clap::DataType::Float32, {3, 4});
    clap::SoftmaxDesc log_softmax;
    log_softmax.axis = -1;
    log_softmax.log_softmax = true;

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forSoftmax(clap::DnnOperation::SoftmaxForward, log_softmax, d),
        device.ctx);
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "LogSoftmax supported: " << (support ? "yes" : "no") << " " << support.reason << "\n";
    if (!support)
        return 0;

    std::vector<float> x = {1, 2, 3, 4, 0, 0, 0, 0, -1, 2, -3, 4};
    std::vector<float> dy = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    void* d_x = device.uploadVector(x);
    void* d_dy = device.uploadVector(dy);
    void* d_y = device.allocate(d.bytes());
    void* d_dx = device.allocate(d.bytes());

    backend->softmaxForward(log_softmax, d, d_x, d, d_y, device.ctx);
    backend->softmaxBackward(log_softmax, d, d_y, d, d_dy, d, d_dx, device.ctx);

    const auto y = device.downloadVector<float>(d_y, x.size());
    const auto dx = device.downloadVector<float>(d_dx, x.size());

    std::cout << "forward : ";
    for (float v : y) std::cout << v << ' ';
    std::cout << "\nbackward: ";
    for (float v : dx) std::cout << v << ' ';
    std::cout << '\n';
}
