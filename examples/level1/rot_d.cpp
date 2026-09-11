#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  double x[2] = {1.0, 2.0};
  double y[2] = {3.0, 4.0};
  auto b = BlasFactory::create();
  b->rot(2, x, 1, y, 1, 0.6, 0.8);
  std::cout << x[0] << " " << y[0] << "\n";
}
