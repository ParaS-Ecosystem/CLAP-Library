#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 3;

    double alpha = 1.0;

    double x[] = {
        1.0,
	2.0,
	3.0
    };

    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    int incx = 1;
    int lda = n;

    auto backend = clap::BlasFactory::create();

    backend->syr(
        Layout::ColMajor,
        Uplo::Upper,
        n,
	alpha,
	x,
	incx,
	A,
	lda
    );

    printf("Updated upper triangle of A:\n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2lf ", A[i + j * lda]);
        }
	printf("\n");
    }

    return 0;
}


