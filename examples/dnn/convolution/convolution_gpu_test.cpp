#include <dnn.hh>

#include <chrono>
#include <iostream>
#include <vector>

int main()
{
    const int N = 2;
    const int C = 16;
    const int H = 64;
    const int W = 64;

    const int K = 32;
    const int R = 3;
    const int S = 3;

    const int pad = 1;
    const int stride = 1;

    const int OH = (H + 2 * pad - R) / stride + 1;
    const int OW = (W + 2 * pad - S) / stride + 1;

    std::vector<float> x(static_cast<std::size_t>(N) * C * H * W, 1.0f);
    std::vector<float> w(static_cast<std::size_t>(K) * C * R * S, 0.01f);
    std::vector<float> y(static_cast<std::size_t>(N) * K * OH * OW, 0.0f);

    clap::TensorDesc x_desc({N, C, H, W});
    clap::TensorDesc w_desc({K, C, R, S});
    clap::TensorDesc y_desc({N, K, OH, OW});

    clap::ConvolutionDesc conv;
    conv.pad_h = pad;
    conv.pad_w = pad;
    conv.stride_h = stride;
    conv.stride_w = stride;
    conv.dilation_h = 1;
    conv.dilation_w = 1;

    auto backend = clap::createDnnBackend();

    std::cout << "[APP] Selected backend: " << backend->name() << "\n";

    const int iterations = 50;
    const auto begin = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        backend->convolutionForward(
            x_desc, x.data(),
            w_desc, w.data(),
            conv,
            y_desc, y.data());
    }

    const auto finish = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(finish - begin).count();

    std::cout << "[APP] Iterations   : " << iterations << "\n";
    std::cout << "[APP] Total time   : " << ms << " ms\n";
    std::cout << "[APP] Average time : " << ms / iterations << " ms\n";
    std::cout << "[APP] y[0]         : " << y[0] << "\n";

    return 0;
}
