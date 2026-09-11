
#include <blas.hh>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
using namespace clap;
using namespace std::chrono;

int main() {
  int n, k;
  printf("Enter matrix size n (float): ");
  scanf("%d", &n);
  printf("Enter number of super-diagonals k: ");
  scanf("%d", &k);
  printf("n = %d, k = %d\n", n, k);
  srand(time(NULL));

  int lda = k + 1;
  float *A = (float *)malloc(lda * n * sizeof(float));
  float *X = (float *)malloc(n * sizeof(float));
  if (A == NULL || X == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < lda * n; i++) A[i] = (float)((rand() % 10) + 1);
  for (int i = 0; i < n; i++) X[i] = (float)((rand() % 10) + 1);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->tbmv(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, n, k, A, lda, X, 1);
  auto stop = high_resolution_clock::now();

  printf("\nAfter TBMV:\n");
  for (int i = 0; i < n; i++) {
    printf("X[%d] = %f\n", i, X[i]);
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(A); free(X);
  return 0;
}
