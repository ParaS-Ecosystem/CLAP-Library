#include <blas.hh>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
using namespace clap;
using namespace std::chrono;

int main() {
  double d1, d2, b1, b2;
  double P[5];

  printf("Enter d1 (double): ");
  scanf("%lf", &d1);
  printf("Enter d2 (double): ");
  scanf("%lf", &d2);
  printf("Enter b1 (double): ");
  scanf("%lf", &b1);
  printf("Enter b2 (double): ");
  scanf("%lf", &b2);

  printf("d1 = %lf, d2 = %lf, b1 = %lf, b2 = %lf\n", d1, d2, b1, b2);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->rotmg(&d1, &d2, &b1, b2, P);
  auto stop = high_resolution_clock::now();

  printf("\nAfter ROTMG:\n");
  printf("d1 = %lf\n", d1);
  printf("d2 = %lf\n", d2);
  printf("b1 = %lf\n", b1);
  for (int i = 0; i < 5; i++) {
    printf("P[%d] = %lf\n", i, P[i]);
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  return 0;
}