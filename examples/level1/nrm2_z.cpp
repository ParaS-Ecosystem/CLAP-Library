#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<double>;
  C x[2] = {C{3, 4}, C{1, -2}};
  auto b = BlasFactory::create();
  std::cout << b->nrm2(2, x, 1) << "\n";
  return 0;
}