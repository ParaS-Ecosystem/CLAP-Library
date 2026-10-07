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

    printf("Enter matrix size n (solves nxn system Ax=b): ");
    scanf("%d", &n);

    printf("n = %d\n", n);

    srand(12345);

    double *A    = (double *)malloc(n * n * sizeof(double));
    double *B    = (double *)malloc(n * sizeof(double));
    double *Aorig= (double *)malloc(n * n * sizeof(double));
    double *Borig= (double *)malloc(n * sizeof(double));
    int    *ipiv = (int    *)malloc(n * sizeof(int));

    if (!A || !B || !Aorig || !Borig || !ipiv) {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            A[i + j * n] = (double)(rand() % 10 + 1);
            if (i != j) row_sum += fabs(A[i + j * n]);
        }
        A[i + i * n] = row_sum + 10.0;
        B[i] = (double)(rand() % 20 + 1);
        Aorig[i * n + 0] = 0.0;
    }

    for (int i = 0; i < n * n; i++) Aorig[i] = A[i];
    for (int i = 0; i < n;     i++) Borig[i] = B[i];

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto start = high_resolution_clock::now();

    backend->dgesv(n, 1, A, n, ipiv, B, n, &info);

    auto end = high_resolution_clock::now();

    if (info != 0) {
        printf("dgesv failed: info = %d\n", info);
        return 1;
    }

    printf("\nSolution x (first 5 elements):\n");
    int show = (n < 5) ? n : 5;
    for (int i = 0; i < show; i++)
        printf("  x[%d] = %.8f\n", i, B[i]);

    double res_norm = 0.0, b_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double ax = 0.0;
        for (int j = 0; j < n; j++)
            ax += Aorig[i + j * n] * B[j];
        double r = ax - Borig[i];
        res_norm += r * r;
        b_norm   += Borig[i] * Borig[i];
    }
    res_norm = sqrt(res_norm);
    b_norm   = sqrt(b_norm);

    printf("\nResidual ||Ax - b|| / ||b|| = %.2e\n", res_norm / b_norm);
    printf("info  = %d  (0 = success)\n", info);

    double elapsed = duration<double>(end - start).count();
    printf("Time  = %.6f seconds\n", elapsed);

    free(A); free(B); free(Aorig); free(Borig); free(ipiv);
    return 0;
}
