
#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int n;

  printf("Enter vector size (float): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  auto total_start = high_resolution_clock::now();

  float *x = (float *)malloc(n * sizeof(float));

  if (x == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  float *y = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    x[i] = (rand() % 10) + 1;

    y[i] = 0.0f;
  }

  auto backend = clap::BlasFactory::create();

  auto start = high_resolution_clock::now();

  backend->copy(n, x, 1, y, 1);

  auto stop = high_resolution_clock::now();

  printf("\nAfter CLAP COPY:\n");

  printf("+-----+-------+-------+\n");

  printf("| i   | x[i]  | y[i]  |\n");

  printf("+-----+-------+-------+\n");

  for (int i = 0; i < n; i++) {
    printf("| %3d | %5.1f | %5.1f |\n", i, x[i], y[i]);
  }

  printf("+-----+-------+-------+\n");

  free(x);
  free(y);

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time      : %.6f s\n", total_seconds);

  printf("============================\n");

  return 0;
}
