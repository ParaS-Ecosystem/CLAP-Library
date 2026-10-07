#include <dnn.hh>

#include <chrono>
#include <iostream>
#include <vector>

int main()
{
    const int N = 2;
    const int C = 32;
    const int H = 64;
    const int W = 64;

    const int K = 64;
    const int R = 3;
    const int S = 3;

    const int padding = 1;
    const int stride = 1;

    const int outputH =
        (H + 2 * padding - R) / stride + 1;

    const int outputW =
        (W + 2 * padding - S) / stride + 1;

    const std::size_t inputElements =
        static_cast<std::size_t>(N) * C * H * W;

    const std::size_t filterElements =
        static_cast<std::size_t>(K) * C * R * S;

    const std::size_t outputElements =
        static_cast<std::size_t>(N) * K * outputH * outputW;

    std::vector<float> input(
        inputElements,
        1.0f
    );

    std::vector<float> filter(
        filterElements,
        0.01f
    );

    std::vector<float> output(
        outputElements,
        0.0f
    );

    clap::TensorDesc inputDesc({
        N,
        C,
        H,
        W
    });

    clap::TensorDesc filterDesc({
        K,
        C,
        R,
        S
    });

    clap::TensorDesc outputDesc({
        N,
        K,
        outputH,
        outputW
    });

    clap::ConvolutionDesc convDesc;

    convDesc.pad_h = padding;
    convDesc.pad_w = padding;

    convDesc.stride_h = stride;
    convDesc.stride_w = stride;

    convDesc.dilation_h = 1;
    convDesc.dilation_w = 1;

    std::cout << "Creating CLAP_DNN backend...\n";

    auto backend =
        clap::createDnnBackend();

    const int iterations = 50;

    std::cout
        << "Running "
        << iterations
        << " convolution iterations...\n";

    const auto start =
        std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        backend->convolutionForward(
            inputDesc,
            input.data(),

            filterDesc,
            filter.data(),

            convDesc,

            outputDesc,
            output.data()
        );

        if ((i + 1) % 10 == 0)
        {
            std::cout
                << "Iteration "
                << i + 1
                << " completed\n";
        }
    }

    const auto end =
        std::chrono::high_resolution_clock::now();

    const double totalMs =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    std::cout
        << "\nTotal time   : "
        << totalMs
        << " ms\n";

    std::cout
        << "Average time : "
        << totalMs / iterations
        << " ms\n";

    std::cout
        << "Output[0]    : "
        << output[0]
        << "\n";

    return 0;
}
