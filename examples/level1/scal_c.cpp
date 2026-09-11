#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C alpha{2, 1};
  auto b = BlasFactory::create();
  b->scal(2, &alpha, x, 1);
  float real_alpha = float(0.5);
  b->scal(2, real_alpha, x, 1);
  std::cout << x[0] << " " << x[1] << "\n";
  return 0;
}