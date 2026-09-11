
#include <blas.hh>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
using namespace clap;
using namespace std::chrono;

int main() {
  int n;
  printf("Enter matrix size n (float): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  int lda = n;
  const float alpha = 2.0f;
  float *X = (float *)malloc(n * sizeof(float));
  float *Y = (float *)malloc(n * sizeof(float));
  float *A = (float *)malloc(lda * n * sizeof(float));
  if (X == NULL || Y == NULL || A == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) { X[i] = (float)((rand() % 10) + 1); Y[i] = (float)((rand() % 10) + 1); }
  for (int i = 0; i < lda * n; i++) A[i] = (float)((rand() % 10) + 1);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->syr2(Layout::ColMajor, Uplo::Upper, n, alpha, X, 1, Y, 1, A, lda);
  auto stop = high_resolution_clock::now();

  printf("\nAfter SYR2:\n");
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      printf("A[%d][%d] = %f\n", i, j, A[j * lda + i]);
    }
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(Y); free(A);
  return 0;
}
