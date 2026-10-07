#include <dnn.hh>
#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap::TensorDesc x_desc({1,1,3,3});
    clap::TensorDesc w_desc({1,1,2,2});
    clap::TensorDesc y_desc({1,1,2,2});
    clap::ConvolutionDesc conv;
    clap::ActivationDesc act;
    act.mode = clap::ActivationMode::ReLU;
    std::vector<float> x = {1,-2,3,4,5,-6,7,8,9};
    std::vector<float> w = {1,1,1,1};
    std::vector<float> bias = {-5};
    std::vector<float> y(4,0.0f);
    backend->fusedConvolutionBiasActivation(x_desc,x.data(),w_desc,w.data(),bias.data(),conv,act,y_desc,y.data());
    std::cout << "Backend: " << backend->name() << "\noutput: ";
    for(float v:y) std::cout << v << ' ';
    std::cout << '\n';
}
