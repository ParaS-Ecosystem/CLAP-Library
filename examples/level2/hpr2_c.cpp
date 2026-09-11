
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, 0}};
  C y[2] = {C{3, -1}, C{1, 1}};
  C AP[3] = {};
  C alpha{1, 1};
  auto b = BlasFactory::create();
  b->hpr2(Layout::ColMajor, Uplo::Upper, 2, &alpha, x, 1, y, 1, AP);
  std::cout << AP[0] << " " << AP[2] << "\n";
  return 0;
}
