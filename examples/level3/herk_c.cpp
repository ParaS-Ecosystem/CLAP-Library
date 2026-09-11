
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C A[4] = {C{1, 1}, C{2, 0}, C{3, -1}, C{4, 1}};
  C Cmat[4] = {};
  float alpha = float(1), beta = float(0);
  auto b = BlasFactory::create();
  b->herk(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, 2, 2, alpha, A, 2, beta, Cmat, 2);
  std::cout << Cmat[0] << " " << Cmat[3] << "\n";
  return 0;
}