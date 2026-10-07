#include <dnn.hh>
#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap::TensorDesc d({1,2,1,4});
    clap::LrnDesc lrn;
    std::vector<float> x(8,1.0f), y(8), dy(8,1.0f), dx(8);
    backend->lrnForward(lrn,d,x.data(),d,y.data());
    backend->lrnBackward(lrn,d,x.data(),d,y.data(),d,dy.data(),d,dx.data());
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "forward: "; for(float v:y) std::cout << v << ' ';
    std::cout << "\nbackward: "; for(float v:dx) std::cout << v << ' ';
    std::cout << '\n';
}
