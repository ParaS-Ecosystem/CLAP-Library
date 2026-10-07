#include "clap/lapack_factory.hpp"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace clap;
using namespace std::chrono;

int main() {
    int m, n;

    printf("Enter matrix rows m: ");
    scanf("%d", &m);
    printf("Enter matrix cols n: ");
    scanf("%d", &n);

    printf("m = %d  n = %d\n", m, n);

    srand(12345);

    int k = (m < n) ? m : n;

    double *A      = (double *)malloc(m * n * sizeof(double));
    double *s      = (double *)malloc(k * sizeof(double));
    double *U      = (double *)malloc(m * m * sizeof(double));
    double *VT     = (double *)malloc(n * n * sizeof(double));
    double *superb = (double *)malloc(k * sizeof(double));

    if (!A || !s || !U || !VT || !superb) {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < m * n; i++)
        A[i] = (double)(rand() % 20 + 1);

    auto backend = LapackFactory::create(BackendType::CPU);

    int info = 0;

    auto start = high_resolution_clock::now();

    backend->dgesvd(Layout::ColMajor,
                    Job::All,
                    Job::All,
                    m, n,
                    A, m,
                    s,
                    U,  m,
                    VT, n,
                    superb,
                    &info);

    auto end = high_resolution_clock::now();

    if (info != 0) {
        printf("dgesvd failed: info = %d\n", info);
        return 1;
    }

    printf("\nSingular values (first 5):\n");
    int show = (k < 5) ? k : 5;
    for (int i = 0; i < show; i++)
        printf("  s[%d] = %.8f\n", i, s[i]);

    printf("info = %d  (0 = success)\n", info);

    double elapsed = duration<double>(end - start).count();
    printf("Time = %.6f seconds\n", elapsed);

    free(A); free(s); free(U); free(VT); free(superb);
    return 0;
}
