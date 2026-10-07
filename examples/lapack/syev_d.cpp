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

    printf("Enter matrix size n (eigenvalues of nxn symmetric matrix): ");
    scanf("%d", &n);

    printf("n = %d\n", n);

    srand(12345);

    double *A = (double *)malloc(n * n * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));

    if (!A || !w) {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double val = (double)(rand() % 10 + 1);
            A[i + j * n] = val;
        }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double avg = 0.5 * (A[i + j * n] + A[j + i * n]);
            A[i + j * n] = avg;
            A[j + i * n] = avg;
        }

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto start = high_resolution_clock::now();

    backend->dsyev(Layout::ColMajor,
                   Job::Vec,
                   Uplo::Lower,
                   n, A, n, w, &info);

    auto end = high_resolution_clock::now();

    if (info != 0) {
        printf("dsyev failed: info = %d\n", info);
        return 1;
    }

    printf("\nEigenvalues (first 5, ascending):\n");
    int show = (n < 5) ? n : 5;
    for (int i = 0; i < show; i++)
        printf("  w[%d] = %.8f\n", i, w[i]);

    printf("  ...\n");
    printf("  w[%d] = %.8f  (largest)\n", n-1, w[n-1]);

    printf("info = %d  (0 = success)\n", info);

    double elapsed = duration<double>(end - start).count();
    printf("Time = %.6f seconds\n", elapsed);

    free(A); free(w);
    return 0;
}
