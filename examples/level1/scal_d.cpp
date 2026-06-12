#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int n;

  printf("Enter vector size (double): ");
  scanf("%d", &n);

  printf("n = %d\n", n);

  srand(time(NULL));

  const double alpha = 2.0;

  double *x = (double *)malloc(n * sizeof(double));

  if (x == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < n; i++) {
    x[i] = (double)((rand() % 10) + 1);
  }

  auto start = high_resolution_clock::now();

  auto backend = clap::BlasFactory::create();

  backend->scal(n, alpha, x, 1);

  auto stop = high_resolution_clock::now();

  printf("\nAfter SCAL:\n");

  for (int i = 0; i < n; i++) {
    printf("x[%d] = %lf\n", i, x[i]);
  }

  double total_seconds = duration<double>(stop - start).count();

  printf("Total time : %.6f s\n", total_seconds);

  free(x);

  return 0;
}
