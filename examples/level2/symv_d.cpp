
#include <blas.hh>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace clap;
using namespace std::chrono;

int main()
{
    int n;

    printf("Enter matrix size n: ");
    scanf("%d", &n);

    const double alpha = 1.0;
    const double beta  = 0.0;

    srand(time(NULL));

    double* A = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));

    if(A == NULL || x == NULL || y == NULL)
    {
        printf("malloc failed\n");
        return 1;
    }

    for(int i = 0; i < n; i++)
    {
        for(int j = i; j < n; j++)
        {
            double val =
                (double)((rand() % 10) + 1);

            A[i + j * n] = val;
            A[j + i * n] = val;
        }
    }

    for(int i = 0; i < n; i++)
    {
        x[i] = (double)((rand() % 10) + 1);
        y[i] = 0.0;
    }

    auto backend =
        clap::BlasFactory::create();

    auto start =
        high_resolution_clock::now();

    backend->symv(Layout::ColMajor,Uplo::Upper,n,alpha,A,n,x,1,beta,y,1
    );

    auto stop =high_resolution_clock::now();

     printf("\nResult y:\n");

    for(int i = 0; i < n; i++)
    {
        printf(
            "y[%d] = %f\n",
            i,
            y[i]
        );
    }

    double total_seconds =
        duration<double>(stop - start).count();

    printf(
        "Total time : %.6f s\n",
        total_seconds
    );

    free(A);
    free(x);
    free(y);

    return 0;
}
