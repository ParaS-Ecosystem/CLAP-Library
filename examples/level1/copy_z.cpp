#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<double>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C y[2] = {};
  auto b = BlasFactory::create();
  b->copy(2, x, 1, y, 1);
  std::cout << y[0] << " " << y[1] << "\n";
  return 0;
}