
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

  const float alpha = 1.0f;
  const float beta = 0.0f;

  srand(time(NULL));

  float *A = (float *)malloc(m * m * sizeof(float));

  float *B = (float *)malloc(m * n * sizeof(float));

  float *C = (float *)malloc(m * n * sizeof(float));

  if (A == NULL || B == NULL || C == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      float val = (float)((rand() % 10) + 1);

      A[i + j * m] = val;
      A[j + i * m] = val;
    }
  }

  for (int i = 0; i < m * n; i++) {
    B[i] = (float)((rand() % 10) + 1);
  }

  for (int i = 0; i < m * n; i++) {
    C[i] = 0.0f;
  }

  auto backend = clap::BlasFactory::create();

  auto start = high_resolution_clock::now();

  backend->symm(Layout::ColMajor, Side::Left, Uplo::Upper, m, n, alpha, A, m, B,
                m, beta, C, m);

  auto stop = high_resolution_clock::now();

  if (m <= 10 && n <= 10) {
    printf("\nSymmetric Matrix A (%d x %d)\n", m, m);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        printf("%8.2f ", A[i + j * m]);
      }
      printf("\n");
    }

    printf("\nMatrix B (%d x %d)\n", m, n);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%8.2f ", B[i + j * m]);
      }
      printf("\n");
    }

    printf("\nResult Matrix C (%d x %d)\n", m, n);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%10.2f ", C[i + j * m]);
      }
      printf("\n");
    }
  } else {
    printf("\nFirst 10 elements of C:\n");

    int limit = (m * n < 10) ? (m * n) : 10;

    for (int i = 0; i < limit; i++) {
      printf("C[%d] = %f\n", i, C[i]);
    }
  }

  double total_seconds = duration<double>(stop - start).count();

  printf("\nTotal time : %.6f s\n", total_seconds);

  free(A);
  free(B);
  free(C);

  return 0;
}
