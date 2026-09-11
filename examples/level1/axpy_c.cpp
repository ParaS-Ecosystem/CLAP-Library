#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C y[2] = {C{3, 0}, C{4, 1}};
  C alpha{2, 1};
  auto b = BlasFactory::create();
  b->axpy(2, &alpha, x, 1, y, 1);
  std::cout << y[0] << " " << y[1] << "\n";
  return 0;
}