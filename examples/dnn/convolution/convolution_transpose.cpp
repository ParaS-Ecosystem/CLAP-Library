#include <dnn.hh>
#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap::TensorDesc x_desc({1,1,2,2});
    clap::TensorDesc w_desc({1,1,2,2});
    clap::TensorDesc y_desc({1,1,3,3});
    clap::ConvolutionDesc conv;
    std::vector<float> x = {1,2,3,4};
    std::vector<float> w = {1,0,0,1};
    std::vector<float> y(9,0.0f);
    backend->convolutionTransposeForward(x_desc,x.data(),w_desc,w.data(),conv,y_desc,y.data());
    std::cout << "Backend: " << backend->name() << "\noutput: ";
    for(float v:y) std::cout << v << ' ';
    std::cout << '\n';
}
