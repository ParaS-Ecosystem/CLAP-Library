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

    printf("Enter n : ");
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

    for(int col = 0; col < n; col++)
    {
        for(int row = 0; row < n; row++)
        {
            if(row <= col)
                A[col * n + row] =
                    (float)((rand() % 10) + 1);
            else
                A[col * n + row] = 0.0f;
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

    backend->trmv(
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

    double total_seconds =
        duration<double>(stop - start).count();

	 printf("\nAfter TRMV:\n");
 
      for(int i = 0; i < n; i++)
      {
          printf(
              "x[%d] = %lf\n",
              i,
              x[i]
         );
     }
    printf(
        "TRMV Float Time : %.6f s\n",
        total_seconds
    );

    free(A);
    free(x);

    return 0;
}