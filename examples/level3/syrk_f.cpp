#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int n, k;

  printf("Enter n : ");
  scanf("%d", &n);

  printf("Enter k : ");
  scanf("%d", &k);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  srand(time(NULL));

  float *A = (float *)malloc(n * k * sizeof(float));

  float *C = (float *)malloc(n * n * sizeof(float));

  if (A == NULL || C == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int i = 0; i < n * k; i++) {
    A[i] = (float)((rand() % 10) + 1);
  }

  for (int i = 0; i < n * n; i++) {
    C[i] = (float)((rand() % 10) + 1);
  }

  auto backend = clap::BlasFactory::create();

  auto start = high_resolution_clock::now();

  backend->syrk(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, n, k, alpha,
                A, n, beta, C, n);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();

  for (int i = 0; i < 10 && i < m * n; i++) {
    printf("C[%d] = %f\n", i, C[i]);
  }
  printf("SYRK Float Time : %.6f s\n", total_seconds);

  free(A);
  free(C);

  return 0;
}
