#include "clap/lapack_factory.hpp"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace clap;
using namespace std::chrono;

int main() {
    int n;

    printf("Enter matrix size n (Cholesky of nxn SPD matrix): ");
    scanf("%d", &n);

    printf("n = %d\n", n);

    srand(12345);

    double *A    = (double *)malloc(n * n * sizeof(double));
    double *B    = (double *)malloc(n * sizeof(double));

    if (!A || !B) {
        printf("malloc failed\n");
        return 1;
    }

    double *L = (double *)calloc(n * n, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            L[i + j * n] = (double)(rand() % 5 + 1);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += L[i + k * n] * L[j + k * n];
            A[i + j * n] = s + (i == j ? n : 0.0);
        }

    for (int i = 0; i < n; i++)
        B[i] = (double)(rand() % 10 + 1);

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto start = high_resolution_clock::now();

    backend->dpotrf(Layout::ColMajor, Uplo::Lower, n, A, n, &info);

    auto end = high_resolution_clock::now();

    if (info != 0) {
        printf("dpotrf failed: info = %d (matrix not positive definite)\n", info);
        return 1;
    }

    printf("dpotrf info = %d  (0 = success)\n", info);
    printf("Cholesky factor L[0][0] = %.8f\n", A[0]);

    backend->dpotrs(Layout::ColMajor, Uplo::Lower, n, 1, A, n, B, n, &info);
    printf("dpotrs info = %d  (0 = success)\n", info);

    printf("\nSolution (first 5 elements):\n");
    int show = (n < 5) ? n : 5;
    for (int i = 0; i < show; i++)
        printf("  x[%d] = %.8f\n", i, B[i]);

    double elapsed = duration<double>(end - start).count();
    printf("dpotrf Time = %.6f seconds\n", elapsed);

    free(A); free(B); free(L);
    return 0;
}
