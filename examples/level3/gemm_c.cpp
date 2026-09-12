#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  using C = std::complex<float>;

  // Column-major 2x2 matrices
  C A[4] = {
      C{1, 0}, C{3, 0},
      C{2, 0}, C{4, 0}
  };

  C B[4] = {
      C{5, 0}, C{7, 0},
      C{6, 0}, C{8, 0}
  };

  C Cmat[4] = {};

  C alpha{1, 0};
  C beta{0, 0};

  auto b = BlasFactory::create();

  b->gemm(
      Layout::ColMajor,
      Transpose::NoTrans,
      Transpose::NoTrans,
      2, 2, 2,
      &alpha,
      A, 2,
      B, 2,
      &beta,
      Cmat, 2);

  std::cout << "Result C:\n";

  std::cout << Cmat[0] << " " << Cmat[2] << "\n";
  std::cout << Cmat[1] << " " << Cmat[3] << "\n";

  return 0;
}