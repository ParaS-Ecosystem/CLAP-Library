
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
  printf("Enter matrix size n (float complex, packed): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  int apsize = n * (n + 1) / 2;
  std::complex<float> alpha(2.0f, 0.0f);
  std::complex<float> beta(1.0f, 0.0f);
  std::complex<float> *Ap = (std::complex<float> *)malloc(apsize * sizeof(std::complex<float>));
  std::complex<float> *X = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
  std::complex<float> *Y = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
  if (Ap == NULL || X == NULL || Y == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < apsize; i++) Ap[i] = std::complex<float>((rand() % 10) + 1, 0.0f);
  for (int i = 0; i < n; i++) { X[i] = std::complex<float>((rand() % 10) + 1, rand() % 5); Y[i] = std::complex<float>((rand() % 10) + 1, rand() % 5); }

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->hpmv(Layout::ColMajor, Uplo::Upper, n, &alpha, Ap, X, 1, &beta, Y, 1);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HPMV:\n");
  for (int i = 0; i < n; i++) {
    printf("Y[%d] = (%f, %f)\n", i, Y[i].real(), Y[i].imag());
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(Ap); free(X); free(Y);
  return 0;
}
