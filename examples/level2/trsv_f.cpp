
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

    float* A =
        (float*)malloc(
            n * n * sizeof(float)
        );

    float* x =
        (float*)malloc(
            n * sizeof(float)
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
                        (float)((rand() % 10) + 5);
                else
                    A[i + j * n] =
                        (float)((rand() % 10) + 1);
            }
            else
            {
                A[i + j * n] = 0.0f;
            }
        }
    }


    for(int i = 0; i < n; i++)
    {
        x[i] =
            (float)((rand() % 10) + 1);
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
            "x[%d] = %f\n",
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
