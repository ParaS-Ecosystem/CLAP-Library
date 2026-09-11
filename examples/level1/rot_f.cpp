#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  float x[2] = {1.0f, 2.0f};
  float y[2] = {3.0f, 4.0f};
  auto b = BlasFactory::create();
  b->rot(2, x, 1, y, 1, 0.6f, 0.8f);
  std::cout << x[0] << " " << y[0] << "\n";
}