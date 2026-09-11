
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
  double *X = (double *)malloc(n * sizeof(double));
  double *Ap = (double *)malloc(apsize * sizeof(double));
  if (X == NULL || Ap == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) X[i] = (double)((rand() % 10) + 1);
  for (int i = 0; i < apsize; i++) Ap[i] = (double)((rand() % 10) + 1);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->spr(Layout::ColMajor, Uplo::Upper, n, alpha, X, 1, Ap);
  auto stop = high_resolution_clock::now();

  printf("\nAfter SPR:\n");
  for (int i = 0; i < apsize; i++) {
    printf("Ap[%d] = %lf\n", i, Ap[i]);
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(Ap);
  return 0;
}
