#include <dnn.hh>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap::TensorDesc d({1,1,2,4});
    clap::DropoutDesc dropout;
    dropout.probability = 0.5f;
    dropout.seed = 42;
    std::vector<float> x = {1,2,3,4,5,6,7,8};
    std::vector<float> y(8), dy(8,1.0f), dx(8);
    std::vector<std::uint8_t> mask(8);
    backend->dropoutForward(dropout,d,x.data(),d,y.data(),mask.data());
    backend->dropoutBackward(dropout,d,dy.data(),mask.data(),d,dx.data());
    std::cout << "Backend: " << backend->name() << "\nforward: ";
    for(float v:y) std::cout << v << ' ';
    std::cout << "\nmask: ";
    for(auto v:mask) std::cout << int(v) << ' ';
    std::cout << "\nbackward: ";
    for(float v:dx) std::cout << v << ' ';
    std::cout << '\n';
}
