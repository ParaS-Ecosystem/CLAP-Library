
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
  float *Ap = (float *)malloc(apsize * sizeof(float));
  float *X = (float *)malloc(n * sizeof(float));
  if (Ap == NULL || X == NULL) {
    printf("malloc failed\n");
    return 1;
  }
  int idx = 0;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j; i++) {
      float val = (float)((rand() % 5) + 1);
      if (i == j) val += 20.0f;
      Ap[idx++] = val;
    }
  }
  for (int i = 0; i < n; i++) X[i] = (float)((rand() % 10) + 1);

  auto backend = clap::BlasFactory::create();
  auto start = high_resolution_clock::now();
  backend->tpsv(Layout::ColMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, n, Ap, X, 1);
  auto stop = high_resolution_clock::now();

  printf("\nAfter TPSV:\n");
  for (int i = 0; i < n; i++) {
    printf("X[%d] = %f\n", i, X[i]);
  }

  double total_seconds = duration<double>(stop - start).count();
  printf("Total time : %.6f s\n", total_seconds);

  free(Ap); free(X);
  return 0;
}
