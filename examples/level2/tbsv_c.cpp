
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C A[4] = {C{0, 0}, C{2, 0}, C{1, 1}, C{3, 0}};
  C x[2] = {C{5, 1}, C{6, 0}};
  auto b = BlasFactory::create();
  b->tbsv(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 2, 1, A, 2, x, 1);
  std::cout << x[0] << " " << x[1] << "\n";
  return 0;
}
