#include <dnn.hh>
#include <iostream>
#include <vector>

int main() {
    auto dnn = clap::createDnnBackend();
    std::cout << "Backend: " << dnn->name() << "\n";

    clap::TensorDesc xdesc({1,1,4,4});
    clap::TensorDesc wdesc({1,1,3,3});
    clap::TensorDesc ydesc({1,1,2,2});
    clap::ConvolutionDesc conv;

    std::vector<float> x(16), w(9,1.0f), y(4,0.0f);
    for (int i=0;i<16;++i) x[i]=float(i+1);

    dnn->convolutionForward(xdesc,x.data(),wdesc,w.data(),conv,ydesc,y.data());
    for (float v: y) std::cout << v << ' ';
    std::cout << '\n';
}
