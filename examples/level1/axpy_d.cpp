
#include <blas.hh>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;

int main() {
  int n;

  printf("Enter vector size (double): ");
  scanf("%d", &n);

  printf("n = %d\n", n);

  srand(time(NULL));

  const double alpha = 2.0;

  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));

  if (x == NULL || y == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < n; i++) {
    x[i] = (double)((rand() % 10) + 1);

    y[i] = (double)((rand() % 10) + 1);
  }

  auto backend = clap::BlasFactory::create();

  backend->axpy(n, alpha, x, 1, y, 1);


  printf("\nAfter AXPY:\n");

  for (int i = 0; i < n; i++) {
    printf("y[%d] = %lf\n", i, y[i]);
  }

  free(x);
  free(y);

  return 0;
}
