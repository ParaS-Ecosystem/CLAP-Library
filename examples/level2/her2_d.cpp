
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
  printf("Enter matrix size n (double complex): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  int lda = n;
  std::complex<double> alpha(2.0, 0.0);
  std::complex<double> *X = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
  std::complex<double> *Y = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
  std::complex<double> *A = (std::complex<double> *)malloc(lda * n * sizeof(std::complex<double>));
  if (X == NULL || Y == NULL || A == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) { X[i] = std::complex<double>((rand() % 10) + 1, rand() % 5); Y[i] = std::complex<double>((rand() % 10) + 1, rand() % 5); }
  for (int i = 0; i < lda * n; i++) A[i] = std::complex<double>((rand() % 10) + 1, 0.0);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->her2(Layout::ColMajor, Uplo::Upper, n, &alpha, X, 1, Y, 1, A, lda);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HER2:\n");
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      printf("A[%d][%d] = (%lf, %lf)\n", i, j, A[j * lda + i].real(), A[j * lda + i].imag());
    }
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(Y); free(A);
  return 0;
}
