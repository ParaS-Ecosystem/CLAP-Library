
#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int m, n;

  printf("Enter rows m: ");
  scanf("%d", &m);

  printf("Enter cols n: ");
  scanf("%d", &n);

  printf("m = %d, n = %d\n", m, n);

  srand(time(NULL));

  const float alpha = 1.0f;

  float *A = (float *)malloc(m * n * sizeof(float));

  float *x = (float *)malloc(m * sizeof(float));

  float *y = (float *)malloc(n * sizeof(float));

  if (A == NULL || x == NULL || y == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < m * n; i++) {
    A[i] = (float)((rand() % 10) + 1);
  }

  for (int i = 0; i < m; i++) {
    x[i] = (float)((rand() % 10) + 1);
  }

  for (int i = 0; i < n; i++) {
    y[i] = (float)((rand() % 10) + 1);
  }

  auto start = high_resolution_clock::now();

  auto backend = clap::BlasFactory::create();

  backend->ger(Layout::ColMajor, m, n, alpha, x, 1, y, 1, A, m);

  auto stop = high_resolution_clock::now();

  if (m <= 10 && n <= 10) {
    printf("\nUpdated Matrix A:\n");

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%8.2f ", A[i + j * m]);
      }
      printf("\n");
    }
  } else {
    printf("\nFirst 10 elements of A:\n");

    int limit = (m * n < 10) ? (m * n) : 10;

    for (int i = 0; i < limit; i++) {
      printf("A[%d] = %f\n", i, A[i]);
    }
  }

  free(A);
  free(x);
  free(y);

  double total_seconds = duration<double>(stop - start).count();

  printf("\nTotal time : %.6f s\n", total_seconds);

  printf("============================\n");

  return 0;
}
