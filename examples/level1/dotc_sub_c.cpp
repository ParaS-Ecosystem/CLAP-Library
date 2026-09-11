#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C y[2] = {C{3, 0}, C{1, 2}};
  C result{};
  auto b = BlasFactory::create();
  b->dotc_sub(2, x, 1, y, 1, &result);
  std::cout << result << "\n";
  return 0;
}