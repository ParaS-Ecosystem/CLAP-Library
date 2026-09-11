
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C A[4] = {C{2, 0}, C{0, 0}, C{1, 1}, C{3, 0}};
  C B[4] = {C{5, 1}, C{6, 0}, C{3, 0}, C{4, 1}};
  C alpha{1, 0};
  auto b = BlasFactory::create();
  b->trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 2, 2, &alpha, A, 2, B, 2);
  std::cout << B[0] << " " << B[3] << "\n";
  return 0;
}
