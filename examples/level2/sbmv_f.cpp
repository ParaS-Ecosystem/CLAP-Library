
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

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    srand(time(NULL));

    int lda = k + 1;

    float* A =
        (float*)malloc(
            lda * n * sizeof(float)
        );

    float* x =
        (float*)malloc(
            n * sizeof(float)
        );

    float* y =
        (float*)malloc(
            n * sizeof(float)
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
                (float)((rand() % 10) + 1);
        }
    }

    for(int i = 0; i < n; i++)
    {
        x[i] =
            (float)((rand() % 10) + 1);

        y[i] = 0.0f;
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
