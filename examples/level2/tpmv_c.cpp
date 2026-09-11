
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C AP[3] = {C{2, 0}, C{1, 1}, C{3, 0}};
  C x[2] = {C{1, 0}, C{2, 1}};
  auto b = BlasFactory::create();
  b->tpmv(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 2, AP, x, 1);
  std::cout << x[0] << " " << x[1] << "\n";
  return 0;
}
