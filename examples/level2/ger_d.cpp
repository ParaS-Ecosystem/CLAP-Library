#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int m = 2;
    int n = 3;

    double alpha = 1.0;

    double x[] = {
        1.0,
	2.0
    };

    double y[] = {
        3.0,
	4.0,
	5.0
    };

    double A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };

    int incx = 1;
    int incy = 1;
    int lda = m;

    auto backend = clap::BlasFactory::create();

    backend->ger(
        Layout::ColMajor,
        m,
	n,
	alpha,
	x,
	incx,
	y,
	incy,
	A,
	lda
    );

    printf("Result matrix A:\n");

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2lf ", A[i + j * lda]);
        }
	printf("\n");
    }

    return 0;
}
