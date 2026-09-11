
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
  std::complex<double> beta(1.0, 0.0);

  std::complex<double> *A = (std::complex<double> *)malloc(lda * n * sizeof(std::complex<double>));
  std::complex<double> *X = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
  std::complex<double> *Y = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
  if (A == NULL || X == NULL || Y == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < lda * n; i++) {
    A[i] = std::complex<double>((rand() % 10) + 1, (rand() % 5));
  }
  for (int i = 0; i < n; i++) {
    X[i] = std::complex<double>((rand() % 10) + 1, 0.0);
    Y[i] = std::complex<double>((rand() % 10) + 1, 0.0);
  }

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->hemv(Layout::ColMajor, Uplo::Upper, n, &alpha, A, lda, X, 1, &beta, Y, 1);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HEMV:\n");
  for (int i = 0; i < n; i++) {
    printf("Y[%d] = (%lf, %lf)\n", i, Y[i].real(), Y[i].imag());
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(A); free(X); free(Y);
  return 0;
}
