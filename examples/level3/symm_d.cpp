
#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int m, n;

  printf("Enter rows m : ");
  scanf("%d", &m);

  printf("Enter cols n : ");
  scanf("%d", &n);

  const double alpha = 1.0;
  const double beta = 0.0;

  srand(time(NULL));

  double *A = (double *)malloc(m * m * sizeof(double));

  double *B = (double *)malloc(m * n * sizeof(double));

  double *C = (double *)malloc(m * n * sizeof(double));

  if (A == NULL || B == NULL || C == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      double val = (double)((rand() % 10) + 1);

      A[i + j * m] = val;
      A[j + i * m] = val;
    }
  }

  for (int i = 0; i < m * n; i++) {
    B[i] = (double)((rand() % 10) + 1);
  }

  for (int i = 0; i < m * n; i++) {
    C[i] = 0.0;
  }

  auto backend = clap::BlasFactory::create();

  auto start = high_resolution_clock::now();

  backend->symm(Layout::ColMajor, Side::Left, Uplo::Upper, m, n, alpha, A, m, B,
                m, beta, C, m);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();

  for (int i = 0; i < 10 && i < m * n; i++) {
    printf("C[%d] = %f\n", i, B[i]);
  }
  printf("Total time : %.6f s\n", total_seconds);

  free(A);
  free(B);
  free(C);

  return 0;
}
