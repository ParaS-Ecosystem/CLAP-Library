
#include <blas.hh>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
using namespace clap;
using namespace std::chrono;

int main() {
  int n;
  printf("Enter matrix size n (double, packed): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  int apsize = n * (n + 1) / 2;
  const double alpha = 2.0;
  const double beta = 1.0;
  double *Ap = (double *)malloc(apsize * sizeof(double));
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  if (Ap == NULL || x == NULL || y == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < apsize; i++) Ap[i] = (double)((rand() % 10) + 1);
  for (int i = 0; i < n; i++) { x[i] = (double)((rand() % 10) + 1); y[i] = (double)((rand() % 10) + 1); }

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->spmv(Layout::ColMajor, Uplo::Upper, n, alpha, Ap, x, 1, beta, y, 1);
  auto stop = high_resolution_clock::now();

  printf("\nAfter SPMV:\n");
  for (int i = 0; i < n; i++) {
    printf("y[%d] = %lf\n", i, y[i]);
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(Ap); free(x); free(y);
  return 0;
}
