
#include <blas.hh>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
using namespace clap;
using namespace std::chrono;

int main() {
  int n;
  printf("Enter matrix size n (float, packed): ");
  scanf("%d", &n);
  printf("n = %d\n", n);
  srand(time(NULL));

  int apsize = n * (n + 1) / 2;
  const float alpha = 2.0f;
  float *X = (float *)malloc(n * sizeof(float));
  float *Y = (float *)malloc(n * sizeof(float));
  float *Ap = (float *)malloc(apsize * sizeof(float));
  if (X == NULL || Y == NULL || Ap == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) { X[i] = (float)((rand() % 10) + 1); Y[i] = (float)((rand() % 10) + 1); }
  for (int i = 0; i < apsize; i++) Ap[i] = (float)((rand() % 10) + 1);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->spr2(Layout::ColMajor, Uplo::Upper, n, alpha, X, 1, Y, 1, Ap);
  auto stop = high_resolution_clock::now();

  printf("\nAfter SPR2:\n");
  for (int i = 0; i < apsize; i++) {
    printf("Ap[%d] = %f\n", i, Ap[i]);
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(X); free(Y); free(Ap);
  return 0;
}
