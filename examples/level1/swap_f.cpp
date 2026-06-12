
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

  float *x = (float *)malloc(n * sizeof(float));

  float *y = (float *)malloc(n * sizeof(float));

  if (x == NULL || y == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < n; i++) {
    x[i] = (float)((rand() % 10) + 1);

    y[i] = (float)((rand() % 10) + 11);
  }

  auto start = high_resolution_clock::now();

  auto backend = clap::BlasFactory::create();
  backend->swap(n, x, 1, y, 1);

  auto stop = high_resolution_clock::now();

  printf("\nAfter SWAP:\n");

  for (int i = 0; i < n; i++) {
    printf("x[%d]=%f  y[%d]=%f\n", i, x[i], i, y[i]);
  }

  double total_seconds = duration<double>(stop - start).count();

  printf("Total time : %.6f s\n", total_seconds);

  free(x);
  free(y);

  return 0;
}
