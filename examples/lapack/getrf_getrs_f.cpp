#include "clap/lapack_factory.hpp"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace clap;
using namespace std::chrono;

int main() {
    int n, nrhs;

    printf("Enter matrix size n: ");
    scanf("%d", &n);
    printf("Enter number of right-hand sides nrhs: ");
    scanf("%d", &nrhs);

    printf("n = %d  nrhs = %d\n", n, nrhs);

    srand(12345);

    float *A    = (float *)malloc(n * n    * sizeof(float));
    float *B    = (float *)malloc(n * nrhs * sizeof(float));
    int   *ipiv = (int   *)malloc(n        * sizeof(int));

    if (!A || !B || !ipiv) {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n * n;    i++) A[i] = (float)(rand() % 10 + 1);
    for (int i = 0; i < n * nrhs; i++) B[i] = (float)(rand() % 10 + 1);

    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++) if (i != j) s += fabsf(A[i + j * n]);
        A[i + i * n] = s + 10.0f;
    }

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto t0 = high_resolution_clock::now();
    backend->sgetrf(Layout::ColMajor, n, n, A, n, ipiv, &info);
    auto t1 = high_resolution_clock::now();

    printf("\nsgetrf info = %d  (0 = success)\n", info);
    if (info != 0) { printf("Factorization failed\n"); return 1; }

    backend->sgetrs(Layout::ColMajor, Transpose::NoTrans,
                    n, nrhs, A, n, ipiv, B, n, &info);
    auto t2 = high_resolution_clock::now();

    printf("sgetrs info = %d  (0 = success)\n", info);

    printf("\nSolution (first RHS, first 5 elements):\n");
    int show = (n < 5) ? n : 5;
    for (int i = 0; i < show; i++)
        printf("  x[%d] = %.6f\n", i, B[i]);

    printf("\nsgetrf time = %.6f sec\n", duration<double>(t1-t0).count());
    printf("sgetrs time = %.6f sec\n", duration<double>(t2-t1).count());
    printf("total  time = %.6f sec\n", duration<double>(t2-t0).count());

    free(A); free(B); free(ipiv);
    return 0;
}
