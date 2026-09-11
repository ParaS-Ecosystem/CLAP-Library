#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, 0}};
  C y[2] = {C{3, 1}, C{1, -1}};
  C A[4] = {};
  C alpha{1, 0};
  auto b = BlasFactory::create();
  b->gerc(Layout::ColMajor, 2, 2, &alpha, x, 1, y, 1, A, 2);
  std::cout << A[0] << " " << A[3] << "\n";
  return 0;
}
