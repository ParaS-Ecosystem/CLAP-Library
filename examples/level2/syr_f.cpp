#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 3;

    float alpha = 1.0f;

    float x[] = {
        1.0f,
	2.0f,
	3.0f
    };

    float A[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
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
            printf("%8.2f ", A[i + j * lda]);
        }
	printf("\n");
    }

    return 0;
}

