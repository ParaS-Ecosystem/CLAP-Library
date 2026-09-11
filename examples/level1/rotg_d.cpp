#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  double a = 3.0, x = 4.0, c = 0.0, s = 0.0;
  auto b = BlasFactory::create();
  b->rotg(&a, &x, &c, &s);
  std::cout << a << " " << c << " " << s << "\n";
}