
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

  const float alpha = 2.0f;
  int apsize = n * (n + 1) / 2;
  std::complex<float> *X = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
  std::complex<float> *AP = (std::complex<float> *)malloc(apsize * sizeof(std::complex<float>));
  if (X == NULL || AP == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) X[i] = std::complex<float>((rand() % 10) + 1, rand() % 5);
  for (int i = 0; i < apsize; i++) AP[i] = std::complex<float>((rand() % 10) + 1, 0.0f);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->hpr(Layout::ColMajor, Uplo::Upper, n, alpha, X, 1, AP);
  auto stop = high_resolution_clock::now();

  printf("\nAfter HPR:\n");
  for (int i = 0; i < apsize; i++) {
    printf("AP[%d] = (%f, %f)\n", i, AP[i].real(), AP[i].imag());
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(AP);
  return 0;
}
