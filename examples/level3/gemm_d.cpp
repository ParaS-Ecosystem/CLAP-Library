
#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int m, n, k;

  printf("Enter m : ");
  scanf("%d", &m);

  printf("Enter n : ");
  scanf("%d", &n);

  printf("Enter k : ");
  scanf("%d", &k);

  const double alpha = 1.0;
  const double beta = 0.0;

  srand(time(NULL));

  double *A = (double *)malloc(m * k * sizeof(double));

  double *B = (double *)malloc(k * n * sizeof(double));

  double *C = (double *)malloc(m * n * sizeof(double));

  if (A == NULL || B == NULL || C == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < m * k; i++) {
    A[i] = (double)((rand() % 10) + 1);
  }

  for (int i = 0; i < k * n; i++) {
    B[i] = (double)((rand() % 10) + 1);
  }

  for (int i = 0; i < m * n; i++) {
    C[i] = 0.0;
  }

  auto backend = clap::BlasFactory::create();

  auto start = high_resolution_clock::now();

  backend->gemm(Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans, m, n,
                k, alpha, A, m, B, k, beta, C, m);

  auto stop = high_resolution_clock::now();

  if (m <= 10 && n <= 10 && k <= 10) {
    printf("\nMatrix A (%d x %d)\n", m, k);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        printf("%8.2lf ", A[i + j * m]);
      }
      printf("\n");
    }

    printf("\nMatrix B (%d x %d)\n", k, n);

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        printf("%8.2lf ", B[i + j * k]);
      }
      printf("\n");
    }

    printf("\nResult Matrix C (%d x %d)\n", m, n);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%10.2lf ", C[i + j * m]);
      }
      printf("\n");
    }
  } else {
    printf("\nFirst 10 elements of C:\n");

    int limit = (m * n < 10) ? (m * n) : 10;

    for (int i = 0; i < limit; i++) {
      printf("C[%d] = %lf\n", i, C[i]);
    }
  }

  double total_seconds = duration<double>(stop - start).count();

  printf("\nTotal time : %.6f s\n", total_seconds);

  free(A);
  free(B);
  free(C);

  return 0;
}
