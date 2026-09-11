
#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;
  C x[2] = {C{1, 1}, C{2, -1}};
  C AP[3] = {};
  float alpha = float(1);
  auto b = BlasFactory::create();
  b->hpr(Layout::ColMajor, Uplo::Upper, 2, alpha, x, 1, AP);
  std::cout << AP[0] << " " << AP[2] << "\n";
  return 0;
}
