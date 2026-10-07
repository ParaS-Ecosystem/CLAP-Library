#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <iostream>
#include <vector>

// Axis-aware softmax on a rank-3 tensor: the last axis (transformer
// attention convention), axis 1, and a strided (transposed) view.
int main() {
    auto backend = clap::createDnnBackend();
    clap_example::Device device(*backend);

    clap::TensorDesc d(clap::DataType::Float32, {2, 3, 4});
    std::vector<float> x(24);
    for (int i = 0; i < 24; ++i)
        x[i] = 0.25f * static_cast<float>(i % 7) - 0.5f;

    void* d_x = device.uploadVector(x);
    void* d_y = device.allocate(d.bytes());

    std::cout << "Backend: " << backend->name() << "\n";

    for (int axis : {-1, 1}) {
        clap::SoftmaxDesc softmax;
        softmax.axis = axis;

        const auto support = backend->supports(
            clap::DnnCapabilityQuery::forSoftmax(clap::DnnOperation::SoftmaxForward, softmax, d),
            device.ctx);
        if (!support) {
            std::cout << "axis " << axis << ": unsupported (" << support.reason << ")\n";
            continue;
        }

        backend->softmaxForward(softmax, d, d_x, d, d_y, device.ctx);
        const auto y = device.downloadVector<float>(d_y, 24);

        std::cout << "axis " << axis << ": ";
        for (float v : y) std::cout << v << ' ';
        std::cout << '\n';
    }

    // The same memory viewed as [2, 4, 3] (dims 1 and 2 swapped through
    // strides); softmax over its last axis normalizes the original axis 1.
    clap::TensorDesc t(clap::DataType::Float32, {2, 4, 3}, {12, 1, 4});
    clap::SoftmaxDesc last;
    last.axis = -1;

    const auto support = backend->supports(
        clap::DnnCapabilityQuery::forSoftmax(clap::DnnOperation::SoftmaxForward, last, t),
        device.ctx);
    if (support) {
        backend->softmaxForward(last, t, d_x, t, d_y, device.ctx);
        const auto y = device.downloadVector<float>(d_y, 24);

        std::cout << "strided view, axis -1: ";
        for (float v : y) std::cout << v << ' ';
        std::cout << '\n';
    } else {
        std::cout << "strided view: unsupported (" << support.reason << ")\n";
    }
}
