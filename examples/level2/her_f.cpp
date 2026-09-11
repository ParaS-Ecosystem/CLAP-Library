
#include <blas.hh>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <complex>
using namespace clap;
using namespace std::chrono;

int main() {
  int n;
  printf("Enter matrix size n (float complex): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  int lda = n;
  const float alpha = 2.0f;
  std::complex<float> *X = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
  std::complex<float> *A = (std::complex<float> *)malloc(lda * n * sizeof(std::complex<float>));
  if (X == NULL || A == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) X[i] = std::complex<float>((rand() % 10) + 1, rand() % 5);
  for (int i = 0; i < lda * n; i++) A[i] = std::complex<float>((rand() % 10) + 1, 0.0f);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->her(Layout::ColMajor, Uplo::Upper, n, alpha, X, 1, A, lda);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HER:\n");
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      printf("A[%d][%d] = (%f, %f)\n", i, j, A[j * lda + i].real(), A[j * lda + i].imag());
    }
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(A);
  return 0;
}
