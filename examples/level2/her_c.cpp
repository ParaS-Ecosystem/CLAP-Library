
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C A[4] = {};
  float alpha = float(1);
  auto b = BlasFactory::create();
  b->her(Layout::ColMajor, Uplo::Upper, 2, alpha, x, 1, A, 2);
  std::cout << A[0] << " " << A[3] << "\n";
  return 0;
}
