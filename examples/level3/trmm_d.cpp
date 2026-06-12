
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

  srand(time(NULL));

  double *A = (double *)malloc(m * m * sizeof(double));

  double *B = (double *)malloc(m * n * sizeof(double));

  if (A == NULL || B == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      if (i <= j) {
        A[i + j * m] = (double)((rand() % 10) + 1);
      } else {
        A[i + j * m] = 0.0;
      }
    }
  }

  for (int i = 0; i < m * n; i++) {
    B[i] = (double)((rand() % 10) + 1);
  }

  auto start = high_resolution_clock::now();

  auto backend = clap::BlasFactory::create();

  backend->trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Transpose::NoTrans,
                Diag::NonUnit, m, n, alpha, A, m, B, m);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();

  for (int i = 0; i < 10 && i < m * n; i++) {
    printf("C[%d] = %f\n", i, B[i]);
  }
  printf("Total time : %.6f s\n", total_seconds);

  free(A);
  free(B);

  return 0;
}
