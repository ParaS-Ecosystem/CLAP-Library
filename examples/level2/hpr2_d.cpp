
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
  printf("Enter matrix size n (double complex, packed): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  std::complex<double> alpha(2.0, 0.0);
  int apsize = n * (n + 1) / 2;
  std::complex<double> *X = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
  std::complex<double> *Y = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
  std::complex<double> *AP = (std::complex<double> *)malloc(apsize * sizeof(std::complex<double>));
  if (X == NULL || Y == NULL || AP == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) { X[i] = std::complex<double>((rand() % 10) + 1, rand() % 5); Y[i] = std::complex<double>((rand() % 10) + 1, rand() % 5); }
  for (int i = 0; i < apsize; i++) AP[i] = std::complex<double>((rand() % 10) + 1, 0.0);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->hpr2(Layout::ColMajor, Uplo::Upper, n, &alpha, X, 1, Y, 1, AP);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HPR2:\n");
  for (int i = 0; i < apsize; i++) {
    printf("AP[%d] = (%lf, %lf)\n", i, AP[i].real(), AP[i].imag());
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(Y); free(AP);
  return 0;
}
