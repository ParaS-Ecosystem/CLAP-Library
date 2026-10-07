#include "clap/lapack_factory.hpp"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace clap;
using namespace std::chrono;

void test_spotri(int n)
{
    printf("\n========================================\n");
    printf("Testing SPOTRI (single precision)\n");
    printf("========================================\n");

    float *A     = (float *)malloc(n * n * sizeof(float));
    float *Aorig = (float *)malloc(n * n * sizeof(float));
    float *M     = (float *)malloc(n * n * sizeof(float));

    if (!A || !Aorig || !M) {
        printf("malloc failed\n");
        free(A);
        free(Aorig);
        free(M);
        return;
    }

    srand(1234);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i + j * n] =
                (float)(rand() % 10 + 1);
        }
    }

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            float sum = 0.0f;

            for (int k = 0; k < n; k++) {
                sum += M[i + k * n] *
                       M[j + k * n];
            }

            if (i == j)
                sum += (float)n;

            A[i + j * n] = sum;
        }
    }

    for (int i = 0; i < n * n; i++)
        Aorig[i] = A[i];

    free(M);

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto start = high_resolution_clock::now();

    backend->spotrf(
        Layout::ColMajor,
        Uplo::Lower,
        n,
        A,
        n,
        &info
    );

    if (info != 0) {
        printf("spotrf failed: info = %d\n", info);

        free(A);
        free(Aorig);
        return;
    }

    backend->spotri(
        Layout::ColMajor,
        Uplo::Lower,
        n,
        A,
        n,
        &info
    );

    auto end = high_resolution_clock::now();

    if (info != 0) {
        printf("spotri failed: info = %d\n", info);

        free(A);
        free(Aorig);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i + j * n] = A[j + i * n];
        }
    }

    printf("\nA^(-1) from SPOTRI:\n");

    int show = (n < 5) ? n : 5;

    for (int i = 0; i < show; i++) {
        for (int j = 0; j < show; j++) {
            printf("  %.6f", A[i + j * n]);
        }
        printf("\n");
    }

    float res_norm = 0.0f;
    float identity_norm = 0.0f;

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            float product = 0.0f;

            for (int k = 0; k < n; k++) {
                product += Aorig[i + k * n] *
                           A[k + j * n];
            }

            float expected =
                (i == j) ? 1.0f : 0.0f;

            float r = product - expected;

            res_norm += r * r;
            identity_norm += expected * expected;
        }
    }

    res_norm = sqrtf(res_norm);
    identity_norm = sqrtf(identity_norm);

    printf("\nSPOTRI residual = %.2e\n",
           res_norm / identity_norm);

    printf("info = %d (0 = success)\n", info);

    double elapsed =
        duration<double>(end - start).count();

    printf("Time = %.6f seconds\n", elapsed);

    free(A);
    free(Aorig);
}

void test_dpotri(int n)
{
    printf("\n========================================\n");
    printf("Testing DPOTRI (double precision)\n");
    printf("========================================\n");

    double *A     = (double *)malloc(n * n * sizeof(double));
    double *Aorig = (double *)malloc(n * n * sizeof(double));
    double *M     = (double *)malloc(n * n * sizeof(double));

    if (!A || !Aorig || !M) {
        printf("malloc failed\n");
        free(A);
        free(Aorig);
        free(M);
        return;
    }

    srand(1234);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i + j * n] =
                (double)(rand() % 10 + 1);
        }
    }

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += M[i + k * n] *
                       M[j + k * n];
            }

            if (i == j)
                sum += (double)n;

            A[i + j * n] = sum;
        }
    }

    for (int i = 0; i < n * n; i++)
        Aorig[i] = A[i];

    free(M);

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto start = high_resolution_clock::now();

    backend->dpotrf(
        Layout::ColMajor,
        Uplo::Lower,
        n,
        A,
        n,
        &info
    );

    if (info != 0) {
        printf("dpotrf failed: info = %d\n", info);

        free(A);
        free(Aorig);
        return;
    }

    backend->dpotri(
        Layout::ColMajor,
        Uplo::Lower,
        n,
        A,
        n,
        &info
    );

    auto end = high_resolution_clock::now();

    if (info != 0) {
        printf("dpotri failed: info = %d\n", info);

        free(A);
        free(Aorig);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i + j * n] = A[j + i * n];
        }
    }

    printf("\nA^(-1) from DPOTRI:\n");

    int show = (n < 5) ? n : 5;

    for (int i = 0; i < show; i++) {
        for (int j = 0; j < show; j++) {
            printf("  %.8f", A[i + j * n]);
        }
        printf("\n");
    }

    double res_norm = 0.0;
    double identity_norm = 0.0;

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            double product = 0.0;

            for (int k = 0; k < n; k++) {
                product += Aorig[i + k * n] *
                           A[k + j * n];
            }

            double expected =
                (i == j) ? 1.0 : 0.0;

            double r = product - expected;

            res_norm += r * r;
            identity_norm += expected * expected;
        }
    }

    res_norm = sqrt(res_norm);
    identity_norm = sqrt(identity_norm);

    printf("\nDPOTRI residual = %.2e\n",
           res_norm / identity_norm);

    printf("info = %d (0 = success)\n", info);

    double elapsed =
        duration<double>(end - start).count();

    printf("Time = %.6f seconds\n", elapsed);

    free(A);
    free(Aorig);
}

int main()
{
    int n;

    printf("Enter matrix size n: ");
    scanf("%d", &n);

    printf("n = %d\n", n);

    test_spotri(n);
    test_dpotri(n);

    return 0;
}
