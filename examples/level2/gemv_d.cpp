
#include <blas.hh>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace clap;
using namespace std::chrono;

int main()
{
    int m, n;

    printf("Enter rows m: ");
    scanf("%d", &m);

    printf("Enter cols n: ");
    scanf("%d", &n);

    printf("m = %d, n = %d\n", m, n);

    srand(time(NULL));

    const double alpha = 1.0;
    const double beta  = 0.0;

    double* A =
        (double*)malloc(
            m * n * sizeof(double)
        );

    double* x =
        (double*)malloc(
            n * sizeof(double)
        );

    double* y =
        (double*)malloc(
            m * sizeof(double)
        );

    if(A == NULL || x == NULL || y == NULL)
    {
        printf("malloc failed\n");
        return 1;
    }

    for(int i = 0; i < m * n; i++)
    {
        A[i] =
            (double)((rand() % 10) + 1);
    }

    for(int i = 0; i < n; i++)
    {
        x[i] =
            (double)((rand() % 10) + 1);
    }

    for(int i = 0; i < m; i++)
    {
        y[i] = 0.0;
    }

    auto start =
        high_resolution_clock::now();

    auto backend =
        clap::BlasFactory::create();

    backend->gemv(Layout::ColMajor,Transpose::NoTrans,m,n,alpha,A,m,x,1, beta,y,1);

    auto stop =
        high_resolution_clock::now();


    printf("\nResult y:\n");

    for(int i = 0; i < m; i++)
    {
        printf(
            "y[%d] = %lf\n",
            i,
            y[i]
        );
    }

    free(A);
    free(x);
    free(y);

    double total_seconds =
        duration<double>(
            stop - start
        ).count();

    printf(
        "Total time : %.6f s\n",
        total_seconds
    );

    printf(
        "============================\n"
    );

    return 0;
}