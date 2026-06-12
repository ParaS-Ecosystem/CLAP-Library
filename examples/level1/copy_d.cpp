
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

  double *x = (double *)malloc(n * sizeof(double));

  if (x == NULL) {
    printf("malloc failed\n");
    return 1;
  }

    double* y =(double*)mallocn * sizeof(double));

    for (int i = 0; i < n; i++) {
      x[i] = (double)((rand() % 10) + 1);

      y[i] = 0.0;
    }

    auto start = high_resolution_clock::now();

    auto backend = clap::BlasFactory::create();

    backend->copy(n, x, 1, y, 1);

    auto stop = high_resolution_clock::now();

    printf("\nAfter CLAP COPY:\n");

    for (int i = 0; i < n; i++) {
      printf("x[%d] = %lf   y[%d] = %lf\n", i, x[i], i, y[i]);
    }

    free(x);
    free(y);

    double total_seconds = duration<double>(stop - start).count();

    printf("Total time : %.6f s\n", total_seconds);

    printf("============================\n");

    return 0;
}
