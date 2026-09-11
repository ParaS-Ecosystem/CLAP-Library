#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<double>;
  C A[6] = {C{0, 0}, C{1, 0}, C{2, 1}, C{3, -1}, C{4, 0}, C{0, 0}};
  C x[2] = {C{1, 0}, C{2, 1}};
  C y[2] = {};
  C alpha{1, 0}, beta{0, 0};
  auto b = BlasFactory::create();
  b->gbmv(Layout::ColMajor, Transpose::NoTrans, 2, 2, 1, 1, &alpha, A, 3, x, 1, &beta, y, 1);
  std::cout << y[0] << " " << y[1] << "\n";
  return 0;
}