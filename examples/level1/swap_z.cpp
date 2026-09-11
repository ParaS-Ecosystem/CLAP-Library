#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<double>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C y[2] = {C{3, 2}, C{4, 1}};
  auto b = BlasFactory::create();
  b->swap(2, x, 1, y, 1);
  std::cout << x[0] << " " << y[0] << "\n";
  return 0;
}