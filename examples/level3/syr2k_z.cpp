
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<double>;
  C A[4] = {C{1, 1}, C{2, 0}, C{3, -1}, C{4, 1}};
  C B[4] = {C{2, 0}, C{1, -1}, C{0, 1}, C{3, 0}};
  C Cmat[4] = {};
  C alpha{1, 0}, beta{0, 0};
  auto b = BlasFactory::create();
  b->syr2k(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, 2, 2, &alpha, A, 2, B, 2, &beta, Cmat, 2);
  std::cout << Cmat[0] << " " << Cmat[3] << "\n";
  return 0;
}
