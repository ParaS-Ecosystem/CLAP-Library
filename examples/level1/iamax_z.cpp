#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<double>;
  C x[3] = {C{1, 1}, C{5, -4}, C{2, 2}};
  auto b = BlasFactory::create();
  std::cout << b->iamax(3, x, 1) << "\n";
  return 0;
}
