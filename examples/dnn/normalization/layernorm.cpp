#include <dnn.hh>
#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap::TensorDesc d({2,1,1,4});
    clap::LayerNormDesc ln;
    std::vector<float> x = {1,2,3,4, 2,4,6,8};
    std::vector<float> scale(4,1.0f), bias(4,0.0f), y(8), mean(2), rstd(2);
    backend->layerNormForward(ln,d,x.data(),scale.data(),bias.data(),d,y.data(),mean.data(),rstd.data());
    std::cout << "Backend: " << backend->name() << "\nforward: ";
    for(float v:y) std::cout << v << ' ';
    std::cout << '\n';
    std::vector<float> dy(8,1.0f), dx(8), dscale(4), dbias(4);
    backend->layerNormBackward(ln,d,x.data(),d,dy.data(),scale.data(),mean.data(),rstd.data(),d,dx.data(),dscale.data(),dbias.data());
    std::cout << "backward dx: ";
    for(float v:dx) std::cout << v << ' ';
    std::cout << '\n';
}
