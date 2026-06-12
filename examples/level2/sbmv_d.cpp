
#include <blas.hh>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace clap;
using namespace std::chrono;

int main()
{
    int n, k;

    printf("Enter matrix size n : ");
    scanf("%d", &n);

    printf("Enter bandwidth k : ");
    scanf("%d", &k);

    const double alpha = 1.0;
    const double beta  = 0.0;

    srand(time(NULL));

    int lda = k + 1;

    double* A =
        (double*)malloc(
            lda * n * sizeof(double)
        );

    double* x =
        (double*)malloc(
            n * sizeof(double)
        );

    double* y =
        (double*)malloc(
            n * sizeof(double)
        );

    if(A == NULL || x == NULL || y == NULL)
    {
        printf("malloc failed\n");
        return 1;
    }

    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < lda; i++)
        {
            A[i + j * lda] =
                (double)((rand() % 10) + 1);
        }
    }

    for(int i = 0; i < n; i++)
    {
        x[i] =
            (double)((rand() % 10) + 1);

        y[i] = 0.0;
    }

    auto backend =
        clap::BlasFactory::create();

    auto start =
        high_resolution_clock::now();

    backend->sbmv(Layout::ColMajor,Uplo::Upper,n,k,alpha,A,lda,x,1,beta,y,1);

    auto stop =
        high_resolution_clock::now();

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
