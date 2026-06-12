
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

    printf("Enter matrix size n : ");
    scanf("%d", &n);

    srand(time(NULL));

    double* A =
        (double*)malloc(
            n * n * sizeof(double)
        );

    double* x =
        (double*)malloc(
            n * sizeof(double)
        );

    if(A == NULL || x == NULL)
    {
        printf("malloc failed\n");
        return 1;
    }

 
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < n; i++)
        {
            if(i <= j)
            {
                if(i == j)
                    A[i + j * n] =
                        (double)((rand() % 10) + 5);
                else
                    A[i + j * n] =
                        (double)((rand() % 10) + 1);
            }
            else
            {
                A[i + j * n] = 0.0;
            }
        }
    }

    for(int i = 0; i < n; i++)
    {
        x[i] =
            (double)((rand() % 10) + 1);
    }

    auto backend =
        clap::BlasFactory::create();

     auto start =
        high_resolution_clock::now();

    backend->trsv(
        Layout::ColMajor,
        Uplo::Upper,
        Transpose::NoTrans,
        Diag::NonUnit,
        n,
        A,
        n,
        x,
        1
    );

    auto stop =
        high_resolution_clock::now();


    printf("\nSolution Vector:\n");

    for(int i = 0; i < n; i++)
    {
        printf(
            "x[%d] = %lf\n",
            i,
            x[i]
        );
    }


    double total_seconds =
        duration<double>(
            stop - start
        ).count();

    printf(
        "Total time : %.6f s\n",
        total_seconds
    );

    free(A);
    free(x);

    return 0;
}
