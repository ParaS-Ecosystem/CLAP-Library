#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int m, n;

  printf("Enter m : ");
  scanf("%d", &m);

  printf("Enter n : ");
  scanf("%d", &n);

  const double alpha = 1.0;

  srand(time(NULL));

  double *A = (double *)malloc(m * m * sizeof(double));

  double *B = (double *)malloc(m * n * sizeof(double));

  if (A == NULL || B == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int col = 0; col < m; col++) {
    for (int row = 0; row < m; row++) {
      if (row <= col)
        A[row + col * m] = (double)((100 % 10) + 1);
      else
        A[row + col * m] = 0.0;
    }
  }

  for (int i = 0; i < m * n; i++) {
    B[i] = (double)(10 + 1);
  }

  auto start = high_resolution_clock::now();

  auto backend = clap::BlasFactory::create();

  backend->trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Transpose::NoTrans,
                Diag::NonUnit, m, n, alpha, A, m, B, m);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();
  for (int i = 0; i < 10 && i < m * n; i++) {
    printf("C[%d] = %f\n", i, B[i]);
  }
  printf("TRSM Double Time : %.6f s\n", total_seconds);

  free(A);
  free(B);

  return 0;
}
