#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int n, k;

  printf("Enter n : ");
  scanf("%d", &n);

  printf("Enter k : ");
  scanf("%d", &k);

  const double alpha = 1.0;
  const double beta = 0.0;

  srand(time(NULL));

  double *A = (double *)malloc(n * k * sizeof(double));

  double *C = (double *)malloc(n * n * sizeof(double));

  if (A == NULL || C == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < n * k; i++) {
    A[i] = (double)((rand() % 10) + 1);
  }

  for (int i = 0; i < n * n; i++) {
    C[i] = (double)((rand() % 10) + 1);
  }

  auto backend = clap::BlasFactory::create();

  auto start = high_resolution_clock::now();

  backend->syrk(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, n, k, alpha,
                A, n, beta, C, n);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();

  for (int i = 0; i < 10 && i < m * n; i++) {
    printf("C[%d] = %f\n", i, C[i]);
  }
  printf("SYRK Double Time : %.6f s\n", total_seconds);

  free(A);
  free(C);

  return 0;
}
