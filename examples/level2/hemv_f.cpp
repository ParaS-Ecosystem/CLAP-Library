
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
  std::complex<float> alpha(2.0f, 0.0f);
  std::complex<float> beta(1.0f, 0.0f);

  std::complex<float> *A = (std::complex<float> *)malloc(lda * n * sizeof(std::complex<float>));
  std::complex<float> *X = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
  std::complex<float> *Y = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
  if (A == NULL || X == NULL || Y == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < lda * n; i++) {
    A[i] = std::complex<float>((rand() % 10) + 1, (rand() % 5));
  }
  for (int i = 0; i < n; i++) {
    X[i] = std::complex<float>((rand() % 10) + 1, 0.0f);
    Y[i] = std::complex<float>((rand() % 10) + 1, 0.0f);
  }

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->hemv(Layout::ColMajor, Uplo::Upper, n, &alpha, A, lda, X, 1, &beta, Y, 1);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HEMV:\n");
  for (int i = 0; i < n; i++) {
    printf("Y[%d] = (%f, %f)\n", i, Y[i].real(), Y[i].imag());
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(A); free(X); free(Y);
  return 0;
}
