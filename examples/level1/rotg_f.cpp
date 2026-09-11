#include <blas.hh>
#include <complex>
#include <iostream>

using namespace clap;

int main() {
  float a = 3.0f, x = 4.0f, c = 0.0f, s = 0.0f;
  auto b = BlasFactory::create();
  b->rotg(&a, &x, &c, &s);
  std::cout << a << " " << c << " " << s << "\n";
}