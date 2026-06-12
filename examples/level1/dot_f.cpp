
#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int n;

  printf("Enter vector size n: ");
  scanf("%d", &n);

  srand(time(NULL));

  float *x = (float *)malloc(n * sizeof(float));

  float *y = (float *)malloc(n * sizeof(float));

  if (x == NULL || y == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < n; i++) {
    x[i] = (float)((rand() % 10) + 1);

    y[i] = (float)((rand() % 10) + 1);
  }

  auto start = high_resolution_clock::now();

  auto backend = clap::BlasFactory::create();

  float result = backend->dot(n, x, 1, y, 1);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();

  printf("Dot Product = %f\n", result);

  printf("Total time : %.6f s\n", total_seconds);

  free(x);
  free(y);

  return 0;
}
