#include <blas.hh>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace clap;
using namespace std::chrono;

int main() {
  int m, n;

  printf("Enter m : ");
  scanf("%d", &m);

  printf("Enter n : ");
  scanf("%d", &n);

  const float alpha = 1.0f;

  srand(time(NULL));

  float *A = (float *)malloc(m * m * sizeof(float));

  float *B = (float *)malloc(m * n * sizeof(float));

  if (A == NULL || B == NULL) {
    printf("malloc failed\n");
    return 1;
  }

  for (int col = 0; col < m; col++) {
    for (int row = 0; row < m; row++) {
      if (row <= col)
        A[row + col * m] = (float)((rand() % 10) + 1);
      else
        A[row + col * m] = 0.0f;
    }
  }

  for (int i = 0; i < m * n; i++) {
    B[i] = (float)((rand() % 10) + 1);
  }

  auto backend = clap::BlasFactory::create();
  
  auto start = high_resolution_clock::now();

  backend->trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Transpose::NoTrans,
                Diag::NonUnit, m, n, alpha, A, m, B, m);

  auto stop = high_resolution_clock::now();

  double total_seconds = duration<double>(stop - start).count();

  for (int i = 0; i < 10 && i < m * n; i++) {
    printf("C[%d] = %f\n", i, B[i]);
  }
  printf("TRSM Float Time : %.6f s\n", total_seconds);

  free(A);
  free(B);

  return 0;
}
